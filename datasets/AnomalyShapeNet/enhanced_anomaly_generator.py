import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.spatial import KDTree
import warnings
warnings.filterwarnings('ignore')


class EnhancedAnomalyGenerator:
    """
    Anomaly generator supporting:
    1. bulge/concavity: RBF smooth deformation
    2. crack: stress-concentration-based linear deformation field
    3. hole: tanh-based depression + point removal
    4. bend: Euler-Bernoulli beam bending
    """

    def __init__(self):
        self.supported_anomaly_types = ['bulge', 'concavity', 'crack', 'hole', 'bend']
        self.anomaly_types = self.supported_anomaly_types

    def generate_pseudo_anomaly_enhanced(self, points, intensity=0.1, region_size=0.05,
                                        anomaly_type=None, **kwargs):
        points = np.array(points, dtype=np.float32)

        if anomaly_type is None:
            anomaly_type = np.random.choice(self.supported_anomaly_types)

        # Smart region selection
        anomaly_center, anomaly_region_indices = self.smart_region_selection_continuous(
            points, region_size, anomaly_type
        )

        # Dispatch to corresponding physics model
        if anomaly_type in ['bulge', 'concavity']:
            anomalous_points, gt_offset = self.generate_rbf_deformation(
                points, anomaly_center, anomaly_region_indices, intensity,
                is_bulge=(anomaly_type == 'bulge'), **kwargs
            )
        elif anomaly_type == 'crack':
            anomalous_points, gt_offset = self.generate_stress_crack(
                points, anomaly_center, anomaly_region_indices, intensity, **kwargs
            )
        elif anomaly_type == 'hole':
            anomalous_points, keep_mask = self.generate_continuous_hole(
                points, anomaly_center, anomaly_region_indices, intensity, **kwargs
            )
            gt_offset = np.zeros_like(points)
            original_kept_points = points[keep_mask]
            gt_offset[keep_mask] = anomalous_points - original_kept_points
            removed_mask = ~keep_mask
            if np.any(removed_mask):
                gt_offset[removed_mask] = np.array([999.0, 999.0, 999.0])
        elif anomaly_type == 'bend':
            anomalous_points, gt_offset = self.generate_bending_deformation(
                points, anomaly_center, anomaly_region_indices, intensity, **kwargs
            )
        else:
            anomalous_points, gt_offset = self.generate_rbf_deformation(
                points, anomaly_center, anomaly_region_indices, intensity, is_bulge=True, **kwargs
            )

        return anomalous_points, gt_offset, anomaly_type

    def smart_region_selection_continuous(self, points, region_size, anomaly_type):
        num_points = len(points)

        anomaly_size_factor = {
            'bulge': 0.1, 'concavity': 0.1, 'hole': 0.1,
            'crack': 0.2, 'bend': 0.1
        }

        target_num = int(num_points * region_size * anomaly_size_factor.get(anomaly_type, 1.0))
        target_num = max(target_num, 100)
        target_num = min(target_num, num_points // 4)

        center_idx = np.random.choice(len(points))
        anomaly_center = points[center_idx]

        region_indices = self._select_connected_region(points, anomaly_center, target_num)

        return anomaly_center, region_indices

    def _select_connected_region(self, points, center, target_num, radius_factor=4.0):
        tree = KDTree(points)

        distances, _ = tree.query(center, k=min(50, len(points)))
        initial_radius = np.mean(distances) * radius_factor

        current_radius = initial_radius * 0.5
        indices = []

        for _ in range(10):
            indices = tree.query_ball_point(center, current_radius)
            if len(indices) >= target_num:
                if len(indices) > target_num:
                    dists = np.linalg.norm(points[indices] - center, axis=1)
                    closest = np.argsort(dists)[:target_num]
                    indices = [indices[j] for j in closest]
                break
            current_radius *= 1.3

        return np.array(indices)

    def generate_rbf_deformation(self, points, center, region_indices, intensity, is_bulge=True, **kwargs):
        points = np.array(points, dtype=np.float32)
        anomalous_points = points.copy()
        gt_offset = np.zeros_like(points)

        if len(region_indices) == 0:
            return anomalous_points, gt_offset

        region_points = points[region_indices]
        region_center = np.mean(region_points, axis=0)
        max_radius = np.max(np.linalg.norm(region_points - region_center, axis=1))

        local_normal = self._estimate_local_normal(region_points)

        base_strength = intensity * 0.5
        direction = 1.0 if is_bulge else -1.0

        # Wendland C2 weights
        all_distances = np.linalg.norm(points - region_center, axis=1)
        normalized_distances = all_distances / max_radius

        weights = np.zeros(len(points))
        valid_mask = normalized_distances <= 1.0
        r = normalized_distances[valid_mask]
        weights[valid_mask] = (1 - r)**4 * (4*r + 1)

        # Apply deformation
        effective_indices = np.where(weights > 1e-4)[0]
        for idx in effective_indices:
            displacement = direction * base_strength * weights[idx] * local_normal
            anomalous_points[idx] += displacement
            gt_offset[idx] = displacement

        return anomalous_points, gt_offset

    def generate_stress_crack(self, points, center, region_indices, intensity, **kwargs):
        """Stress-concentration-based crack simulation."""
        points = np.array(points, dtype=np.float32)
        anomalous_points = points.copy()
        gt_offset = np.zeros_like(points)

        if len(region_indices) == 0:
            return anomalous_points, gt_offset

        region_points = points[region_indices]

        # PCA to determine crack direction
        pca = PCA(n_components=2)
        pca.fit(region_points - center)
        crack_direction = pca.components_[0]
        crack_normal = pca.components_[1]

        crack_length = np.max(np.dot(region_points - center, crack_direction))
        crack_width_base = intensity * 0.1

        for idx in region_indices:
            point = points[idx]
            local_pos = point - center

            s = np.dot(local_pos, crack_direction)
            n = np.dot(local_pos, crack_normal)

            r = np.sqrt(s**2 + n**2) + 1e-6
            stress_intensity = intensity / np.sqrt(r + crack_length * 0.1)

            if abs(s) < crack_length and abs(n) < crack_width_base * 3:
                opening = stress_intensity * crack_width_base * np.exp(-abs(n) / crack_width_base)
                displacement = opening * np.sign(n) * crack_normal
                shear = stress_intensity * 0.3 * crack_width_base * (s / crack_length)
                displacement += shear * crack_direction

                anomalous_points[idx] += displacement
                gt_offset[idx] = displacement

        # Lightweight smoothing
        affected_indices = np.where(np.linalg.norm(gt_offset, axis=1) > 1e-6)[0]
        if len(affected_indices) > 0:
            anomalous_points = self._apply_smoothing(anomalous_points, affected_indices, points, rounds=2)

        return anomalous_points, gt_offset

    def generate_continuous_hole(self, points, center, region_indices, intensity, **kwargs):
        """Tanh-based hole generation: surrounding depression + center removal."""
        points = np.array(points, dtype=np.float32)
        anomalous_points = points.copy()

        if len(region_indices) == 0:
            return points, np.ones(len(points), dtype=bool)

        region_points = points[region_indices]
        region_center = np.mean(region_points, axis=0)
        local_normal = self._estimate_local_normal(region_points)

        max_radius = np.max(np.linalg.norm(region_points - region_center, axis=1))
        dent_depth = intensity * 0.4
        transition_width = max_radius * 0.3
        hole_radius = max_radius * (0.5 + 0.3 * intensity)

        keep_mask = np.ones(len(points), dtype=bool)

        # Step 1: tanh depression
        for idx in region_indices:
            point = points[idx]
            r = np.linalg.norm(point - region_center)

            smooth_factor = np.tanh((max_radius - r) / transition_width)
            depth_factor = smooth_factor * (1 - (r / max_radius)**2)
            displacement = -dent_depth * depth_factor * local_normal
            anomalous_points[idx] += displacement

        # Step 2: center removal
        for idx in region_indices:
            r = np.linalg.norm(points[idx] - region_center)
            if r <= hole_radius:
                keep_mask[idx] = False
            elif r <= hole_radius * 1.3:
                removal_prob = max(0, (hole_radius * 1.3 - r) / (hole_radius * 0.3))
                removal_prob *= (0.6 + 0.4 * np.random.random())
                if np.random.random() < removal_prob:
                    keep_mask[idx] = False

        final_points = anomalous_points[keep_mask]
        return final_points, keep_mask

    def generate_bending_deformation(self, points, center, region_indices, intensity, **kwargs):
        """Euler-Bernoulli beam bending model."""
        points = np.array(points, dtype=np.float32)
        anomalous_points = points.copy()
        gt_offset = np.zeros_like(points)

        if len(region_indices) == 0:
            return anomalous_points, gt_offset

        region_points = points[region_indices]

        pca = PCA(n_components=3)
        pca.fit(region_points - center)
        beam_axis = pca.components_[0]
        bend_direction = pca.components_[1]

        beam_projections = np.dot(region_points - center, beam_axis)
        beam_length = np.max(beam_projections) - np.min(beam_projections)

        if beam_length < 1e-6:
            return anomalous_points, gt_offset

        base_deflection = intensity * beam_length * 0.8
        max_region_distance = np.max(np.linalg.norm(region_points - center, axis=1))
        influence_radius = max_region_distance * 1.5

        all_distances = np.linalg.norm(points - center, axis=1)
        affected_indices = np.where(all_distances <= influence_radius)[0]

        for idx in affected_indices:
            local_pos = points[idx] - center
            x_beam = np.dot(local_pos, beam_axis)
            y_beam = np.dot(local_pos, bend_direction)

            distance_to_center = np.linalg.norm(local_pos)
            if distance_to_center <= max_region_distance:
                weight = 1.0
            elif distance_to_center <= influence_radius:
                weight = ((influence_radius - distance_to_center) / (influence_radius - max_region_distance))**2
            else:
                continue

            # Cantilever beam deflection
            normalized_x = (x_beam - np.min(beam_projections)) / beam_length
            normalized_x = np.clip(normalized_x, 0, 1)
            deflection_factor = normalized_x**2 * (3 - 2*normalized_x)

            deflection = deflection_factor * base_deflection * weight
            cross_factor = np.exp(-abs(y_beam) / (beam_length * 0.5))
            deflection *= cross_factor

            primary_displacement = deflection * bend_direction
            axial_strain = y_beam * deflection / (beam_length**2) * 0.2 if deflection != 0 else 0
            axial_displacement = axial_strain * beam_axis

            total_displacement = primary_displacement + axial_displacement
            anomalous_points[idx] += total_displacement
            gt_offset[idx] = total_displacement

        return anomalous_points, gt_offset

    def _estimate_local_normal(self, region_points):
        """Estimate local normal via PCA."""
        if len(region_points) < 3:
            return np.array([0, 0, 1], dtype=np.float32)

        center = np.mean(region_points, axis=0)
        centered_points = region_points - center

        try:
            pca = PCA(n_components=3)
            pca.fit(centered_points)
            normal = pca.components_[-1]

            if normal[2] < 0:
                normal = -normal

            normal = normal / np.linalg.norm(normal)
            return normal.astype(np.float32)
        except Exception:
            return np.array([0, 0, 1], dtype=np.float32)

    def _apply_smoothing(self, points, modified_indices, original_points, rounds=2, strength=0.2):
        """Lightweight neighbor smoothing."""
        if len(modified_indices) == 0:
            return points

        smoothed_points = points.copy()
        nbrs = NearestNeighbors(n_neighbors=6, algorithm='kd_tree').fit(original_points)

        for round_idx in range(rounds):
            current_strength = strength * (0.7 ** round_idx)

            for idx in modified_indices:
                distances, neighbor_indices = nbrs.kneighbors([original_points[idx]])
                neighbor_indices = neighbor_indices[0][1:]
                neighbor_distances = distances[0][1:]

                if len(neighbor_indices) > 0:
                    neighbor_points = smoothed_points[neighbor_indices]
                    weights = 1.0 / (neighbor_distances + 1e-8)
                    weights = weights / np.sum(weights)
                    weighted_average = np.sum(neighbor_points * weights[:, np.newaxis], axis=0)

                    proposed_pos = (1 - current_strength) * smoothed_points[idx] + \
                                 current_strength * weighted_average

                    # Constrain max deviation
                    original_pos = original_points[idx]
                    deviation = np.linalg.norm(proposed_pos - original_pos)
                    if deviation > 0.03:
                        proposed_pos = original_pos + (proposed_pos - original_pos) / deviation * 0.03

                    smoothed_points[idx] = proposed_pos

        return smoothed_points
