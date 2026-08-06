"""JSON schemas used for T2V / I2V structured PE outputs."""

from __future__ import annotations

from typing import Any


T2V_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "global_layer": {
            "type": "object",
            "properties": {
                "context": {
                    "type": "string"
                },
                "description": {
                    "type": "string"
                },
                "aesthetics": {
                    "type": "object",
                    "properties": {
                        "style": {
                            "type": "string"
                        },
                        "contrast": {
                            "type": "string"
                        },
                        "saturation": {
                            "type": "string"
                        },
                        "color_scheme": {
                            "type": "string"
                        },
                        "visual_effects": {
                            "type": "string"
                        },
                        "mood_atmosphere": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "style",
                        "contrast",
                        "saturation",
                        "color_scheme",
                        "visual_effects",
                        "mood_atmosphere"
                    ],
                    "additionalProperties": False
                },
                "audio_baseline": {
                    "type": "object",
                    "properties": {
                        "ambience": {
                            "type": "string"
                        },
                        "dialogue": {
                            "type": "object",
                            "properties": {
                                "language": {
                                    "type": "string"
                                },
                                "speaker_tags": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                }
                            },
                            "required": [
                                "language",
                                "speaker_tags"
                            ],
                            "additionalProperties": False
                        }
                    },
                    "required": [
                        "ambience",
                        "dialogue"
                    ],
                    "additionalProperties": False
                },
                "objects_static": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "object_id": {
                                "type": "string"
                            },
                            "description": {
                                "type": "string"
                            },
                            "shape_and_color": {
                                "type": "string"
                            },
                            "texture": {
                                "type": "string"
                            },
                            "relative_size": {
                                "type": "string"
                            },
                            "position": {
                                "type": "string"
                            },
                            "orientation": {
                                "type": "string"
                            }
                        },
                        "required": [
                            "object_id",
                            "description",
                            "shape_and_color",
                            "texture",
                            "relative_size",
                            "position",
                            "orientation"
                        ],
                        "additionalProperties": False
                    }
                },
                "video_metadata": {
                    "type": "object",
                    "properties": {
                        "duration": {
                            "type": "number"
                        },
                        "aspect_ratio": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "duration",
                        "aspect_ratio"
                    ],
                    "additionalProperties": False
                },
                "lighting_baseline": {
                    "type": "object",
                    "properties": {
                        "conditions": {
                            "type": "string"
                        },
                        "direction": {
                            "type": "string"
                        },
                        "shadows": {
                            "type": "string"
                        },
                        "source_consistency": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "conditions",
                        "direction",
                        "shadows",
                        "source_consistency"
                    ],
                    "additionalProperties": False
                },
                "environment_baseline": {
                    "type": "object",
                    "properties": {
                        "background_setting": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "background_setting"
                    ],
                    "additionalProperties": False
                },
                "alive_subjects_static": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject_id": {
                                "type": "string"
                            },
                            "description": {
                                "type": "string"
                            },
                            "position": {
                                "type": "string"
                            },
                            "orientation": {
                                "type": "string"
                            },
                            "visual_attributes": {
                                "type": "object",
                                "properties": {
                                    "gender": {
                                        "type": "string"
                                    },
                                    "ethnicity": {
                                        "type": "string"
                                    },
                                    "age_group": {
                                        "type": "string"
                                    },
                                    "facial_features": {
                                        "type": "string"
                                    },
                                    "clothing": {
                                        "type": "string"
                                    },
                                    "appearance_details": {
                                        "type": "string"
                                    }
                                },
                                "required": [
                                    "gender",
                                    "ethnicity",
                                    "age_group",
                                    "facial_features",
                                    "clothing",
                                    "appearance_details"
                                ],
                                "additionalProperties": False
                            }
                        },
                        "required": [
                            "subject_id",
                            "description",
                            "position",
                            "orientation",
                            "visual_attributes"
                        ],
                        "additionalProperties": False
                    }
                },
                "camera_cinematography": {
                    "type": "object",
                    "properties": {
                        "overall_camera_style": {
                            "type": "string"
                        },
                        "default_depth_of_field": {
                            "type": "string"
                        },
                        "lens_focal_length": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "overall_camera_style",
                        "default_depth_of_field",
                        "lens_focal_length"
                    ],
                    "additionalProperties": False
                }
            },
            "required": [
                "context",
                "description",
                "aesthetics",
                "audio_baseline",
                "objects_static",
                "video_metadata",
                "lighting_baseline",
                "environment_baseline",
                "alive_subjects_static",
                "camera_cinematography"
            ],
            "additionalProperties": False
        },
        "dynamic_layer": {
            "type": "object",
            "properties": {
                "timeline_segments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "segment_basic_info": {
                                "type": "object",
                                "properties": {
                                    "timestamp_range": {
                                        "type": "string"
                                    },
                                    "segment_description": {
                                        "type": "string"
                                    },
                                    "active_background": {
                                        "type": "string"
                                    },
                                    "lighting_delta": {
                                        "type": "string"
                                    },
                                    "camera_delta": {
                                        "type": "object",
                                        "properties": {
                                            "focus": {
                                                "type": "string"
                                            },
                                            "camera_position": {
                                                "type": "string"
                                            },
                                            "composition": {
                                                "type": "string"
                                            },
                                            "camera_movement": {
                                                "type": "object",
                                                "properties": {
                                                    "type": {
                                                        "type": "string"
                                                    },
                                                    "direction": {
                                                        "type": "string"
                                                    },
                                                    "intensity": {
                                                        "type": "string"
                                                    }
                                                },
                                                "required": [
                                                    "type",
                                                    "direction",
                                                    "intensity"
                                                ],
                                                "additionalProperties": False
                                            }
                                        },
                                        "required": [
                                            "focus",
                                            "camera_position",
                                            "composition",
                                            "camera_movement"
                                        ],
                                        "additionalProperties": False
                                    }
                                },
                                "required": [
                                    "timestamp_range",
                                    "segment_description",
                                    "active_background",
                                    "lighting_delta",
                                    "camera_delta"
                                ],
                                "additionalProperties": False
                            },
                            "objects": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "object_id": {
                                            "type": "string"
                                        },
                                        "timestamp": {
                                            "type": "string"
                                        },
                                        "number_of_objects": {
                                            "type": "integer"
                                        },
                                        "dynamic_state": {
                                            "type": "object",
                                            "properties": {
                                                "state_change": {
                                                    "type": "string"
                                                },
                                                "motion_detail": {
                                                    "type": "string"
                                                },
                                                "visible_condition": {
                                                    "type": "string"
                                                }
                                            },
                                            "required": [
                                                "state_change",
                                                "motion_detail",
                                                "visible_condition"
                                            ],
                                            "additionalProperties": False
                                        },
                                        "spatial_position": {
                                            "type": "object",
                                            "properties": {
                                                "location": {
                                                    "type": "string"
                                                },
                                                "relative_size": {
                                                    "type": "string"
                                                },
                                                "orientation": {
                                                    "type": "string"
                                                }
                                            },
                                            "required": [
                                                "location",
                                                "relative_size",
                                                "orientation"
                                            ],
                                            "additionalProperties": False
                                        }
                                    },
                                    "required": [
                                        "object_id",
                                        "timestamp",
                                        "number_of_objects",
                                        "dynamic_state",
                                        "spatial_position"
                                    ],
                                    "additionalProperties": False
                                }
                            },
                            "alive_subjects": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "subject_id": {
                                            "type": "string"
                                        },
                                        "timestamp": {
                                            "type": "string"
                                        },
                                        "number_of_subjects": {
                                            "type": "integer"
                                        },
                                        "spatial_position": {
                                            "type": "object",
                                            "properties": {
                                                "location": {
                                                    "type": "string"
                                                },
                                                "relative_size": {
                                                    "type": "string"
                                                },
                                                "orientation": {
                                                    "type": "string"
                                                }
                                            },
                                            "required": [
                                                "location",
                                                "relative_size",
                                                "orientation"
                                            ],
                                            "additionalProperties": False
                                        },
                                        "action": {
                                            "type": "object",
                                            "properties": {
                                                "primary_action": {
                                                    "type": "string"
                                                },
                                                "body_configuration": {
                                                    "type": "string"
                                                },
                                                "motion_detail": {
                                                    "type": "string"
                                                },
                                                "interaction": {
                                                    "type": "string"
                                                },
                                                "facial_expression": {
                                                    "type": "string"
                                                }
                                            },
                                            "required": [
                                                "primary_action",
                                                "body_configuration",
                                                "motion_detail",
                                                "interaction",
                                                "facial_expression"
                                            ],
                                            "additionalProperties": False
                                        }
                                    },
                                    "required": [
                                        "subject_id",
                                        "timestamp",
                                        "number_of_subjects",
                                        "spatial_position",
                                        "action"
                                    ],
                                    "additionalProperties": False
                                }
                            },
                            "text_render": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "text": {
                                            "type": "string"
                                        },
                                        "timestamp": {
                                            "type": "string"
                                        },
                                        "location": {
                                            "type": "string"
                                        },
                                        "size": {
                                            "type": "string"
                                        },
                                        "color": {
                                            "type": "string"
                                        },
                                        "font": {
                                            "type": "string"
                                        },
                                        "appearance_details": {
                                            "type": "string"
                                        }
                                    },
                                    "required": [
                                        "text",
                                        "timestamp",
                                        "location",
                                        "size",
                                        "color",
                                        "font",
                                        "appearance_details"
                                    ],
                                    "additionalProperties": False
                                }
                            },
                            "causal_events": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "event_id": {
                                            "type": "string"
                                        },
                                        "timestamp_range": {
                                            "type": "string"
                                        },
                                        "trigger": {
                                            "type": "string"
                                        },
                                        "effect_outcome": {
                                            "type": "string"
                                        },
                                        "physics": {
                                            "type": "string"
                                        }
                                    },
                                    "required": [
                                        "event_id",
                                        "timestamp_range",
                                        "trigger",
                                        "effect_outcome",
                                        "physics"
                                    ],
                                    "additionalProperties": False
                                }
                            },
                            "audio": {
                                "type": "object",
                                "properties": {
                                    "dialogue_lines": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "timestamp": {
                                                    "type": "string"
                                                },
                                                "speaker": {
                                                    "type": "string"
                                                },
                                                "text": {
                                                    "type": "string"
                                                },
                                                "delivery": {
                                                    "type": "string"
                                                }
                                            },
                                            "required": [
                                                "timestamp",
                                                "speaker",
                                                "text",
                                                "delivery"
                                            ],
                                            "additionalProperties": False
                                        }
                                    },
                                    "ambience_deltas": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "timestamp_range": {
                                                    "type": "string"
                                                },
                                                "ambience": {
                                                    "type": "string"
                                                }
                                            },
                                            "required": [
                                                "timestamp_range",
                                                "ambience"
                                            ],
                                            "additionalProperties": False
                                        }
                                    },
                                    "special_audio_events": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "timestamp": {
                                                    "type": "string"
                                                },
                                                "sound_description": {
                                                    "type": "string"
                                                },
                                                "visual_sync": {
                                                    "type": "string"
                                                }
                                            },
                                            "required": [
                                                "timestamp",
                                                "sound_description",
                                                "visual_sync"
                                            ],
                                            "additionalProperties": False
                                        }
                                    }
                                },
                                "required": [
                                    "dialogue_lines",
                                    "ambience_deltas",
                                    "special_audio_events"
                                ],
                                "additionalProperties": False
                            }
                        },
                        "required": [
                            "segment_basic_info",
                            "objects",
                            "alive_subjects",
                            "text_render",
                            "causal_events",
                            "audio"
                        ],
                        "additionalProperties": False
                    }
                }
            },
            "required": [
                "timeline_segments"
            ],
            "additionalProperties": False
        }
    },
    "required": [
        "global_layer",
        "dynamic_layer"
    ],
    "additionalProperties": False
}


I2V_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "global_caption": {
            "type": "object",
            "properties": {
                "core_caption": {
                    "type": "string"
                },
                "format_and_structure": {
                    "type": "string"
                },
                "narrative_and_emotional_design": {
                    "type": "string"
                },
                "visual_design": {
                    "type": "string"
                },
                "audio_design": {
                    "type": "string"
                },
                "continuity_requirements": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                }
            },
            "required": [
                "core_caption",
                "format_and_structure",
                "narrative_and_emotional_design",
                "visual_design",
                "audio_design",
                "continuity_requirements"
            ],
            "additionalProperties": False
        },
        "reference_bank": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "reference_identifier": {
                        "type": "string"
                    },
                    "reference_type": {
                        "type": "string"
                    },
                    "label": {
                        "type": "string"
                    },
                    "role": {
                        "type": "string"
                    },
                    "description": {
                        "type": "string"
                    },
                    "appearance_or_design": {
                        "type": "string"
                    }
                },
                "required": [
                    "reference_identifier",
                    "reference_type",
                    "label",
                    "role",
                    "description",
                    "appearance_or_design"
                ],
                "additionalProperties": False
            }
        },
        "shot_timeline": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "shot_identifier": {
                        "type": "string"
                    },
                    "time_range_seconds": {
                        "type": "array",
                        "items": {
                            "type": "number"
                        }
                    },
                    "scene_reference_identifier": {
                        "type": "string"
                    },
                    "shot_caption": {
                        "type": "string"
                    },
                    "camera_lighting_and_composition": {
                        "type": "string"
                    },
                    "motion_and_physical_logic": {
                        "type": "string"
                    },
                    "reference_identifiers": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },
                    "audio_event_identifiers": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    }
                },
                "required": [
                    "shot_identifier",
                    "time_range_seconds",
                    "scene_reference_identifier",
                    "shot_caption",
                    "camera_lighting_and_composition",
                    "motion_and_physical_logic",
                    "reference_identifiers",
                    "audio_event_identifiers"
                ],
                "additionalProperties": False
            }
        },
        "audio_event_timeline": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "audio_event_identifier": {
                        "type": "string"
                    },
                    "event_type": {
                        "type": "string"
                    },
                    "time_range_seconds": {
                        "type": "array",
                        "items": {
                            "type": "number"
                        }
                    },
                    "source_reference_identifier": {
                        "type": "string"
                    },
                    "audio_caption": {
                        "type": "string"
                    },
                    "spoken_text": {
                        "type": "string"
                    },
                    "synchronization_note": {
                        "type": "string"
                    }
                },
                "required": [
                    "audio_event_identifier",
                    "event_type",
                    "time_range_seconds",
                    "source_reference_identifier",
                    "audio_caption",
                    "spoken_text",
                    "synchronization_note"
                ],
                "additionalProperties": False
            }
        },
        "visible_text": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "visible_text_identifier": {
                        "type": "string"
                    },
                    "text": {
                        "type": "string"
                    },
                    "time_range_seconds": {
                        "type": "array",
                        "items": {
                            "type": "number"
                        }
                    },
                    "position_and_role": {
                        "type": "string"
                    },
                    "source_reference_identifier": {
                        "type": "string"
                    }
                },
                "required": [
                    "visible_text_identifier",
                    "text",
                    "time_range_seconds",
                    "position_and_role",
                    "source_reference_identifier"
                ],
                "additionalProperties": False
            }
        },
        "generation_requirements": {
            "type": "array",
            "items": {
                "type": "string"
            }
        }
    },
    "required": [
        "global_caption",
        "reference_bank",
        "shot_timeline",
        "audio_event_timeline",
        "visible_text",
        "generation_requirements"
    ],
    "additionalProperties": False
}
