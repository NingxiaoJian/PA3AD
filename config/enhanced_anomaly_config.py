"""
Enhanced anomaly generation config.
Supported anomaly types: bulge, concavity, hole, crack, bend
"""

ENHANCED_ANOMALY_CONFIG = {
    # Basic settings
    'use_enhanced_anomaly': True,
    'anomaly_region_size': 0.05,
    'anomaly_intensity_range': [0.08, 0.24],

    # Anomaly type weights
    'anomaly_type_weights': {
        'bulge': 0.35,
        'concavity': 0.35,
        'hole': 0.10,
        'crack': 0.10,
        'bend': 0.10
    },

    # Detailed params for each anomaly type
    'anomaly_params': {
        'bulge': {
            'intensity_range': [0.08, 0.15],
            'smooth_factor': 2.0,
            'noise_level': 0.1
        },
        'concavity': {
            'intensity_range': [0.06, 0.12],
            'smooth_factor': 2.0,
            'noise_level': 0.1
        },
        'hole': {
            'hole_ratio': [0.7, 0.9],
            'edge_smoothing': True,
            'min_hole_size': 0.02
        },
        'crack': {
            'crack_width': [0.02, 0.04],
            'crack_depth': [0.05, 0.08],
            'crack_roughness': 0.2,
            'branch_probability': 0.3
        },
        'bend': {
            'intensity_range': [0.20, 0.40],
            'base_deflection_factor': 0.8,
            'influence_radius_factor': 1.5,
            'cross_section_decay': 0.5,
            'axial_strain_factor': 0.2,
            'beam_model': 'cantilever'
        }
    },

    # Geometric feature params
    'geometric_features': {
        'neighbor_k': 10,
        'curvature_method': 'variance',
        'edge_detection_threshold': 0.1,
        'local_density_radius': 0.02
    },

    # Smart region selection params
    'region_selection': {
        'selection_method': 'geometric_aware',
        'connectivity_radius_factor': 1.5,
        'score_noise_factor': 0.1,
        'min_region_size': 0.02,
        'max_region_size': 0.25
    },
}


def get_enhanced_anomaly_config():
    return ENHANCED_ANOMALY_CONFIG.copy()


def apply_enhanced_anomaly_config_to_cfg(cfg):
    enhanced_config = get_enhanced_anomaly_config()

    cfg.use_enhanced_anomaly = enhanced_config['use_enhanced_anomaly']
    cfg.anomaly_region_size = enhanced_config['anomaly_region_size']
    cfg.anomaly_intensity_range = enhanced_config['anomaly_intensity_range']

    cfg.enhanced_anomaly_config = enhanced_config

    return cfg
