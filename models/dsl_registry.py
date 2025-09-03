"""
Canonical DSL Operation Registry
Unifies DSL operations across all modules to prevent drift between DSL, PolicyNet, ValueNet, etc.
"""

# Canonical DSL operation registry
DSL_OPS = [
    "rotate90", "rotate180", "rotate270", "flip_h", "flip_v",
    "color_map", "crop_bbox", "flood_fill", "outline", "symmetry",
    "translate", "scale", "tile", "paste", "tile_pattern",
    "crop_nonzero", "extract_color", "resize_nn", "center_pad_to", "identity",
    "count_objects", "count_colors", "arithmetic_op", "find_pattern", "extract_pattern",
    "match_template", "apply_rule", "conditional_map", "grid_union", "grid_intersection",
    "grid_xor", "grid_difference", "flood_select", "select_by_property", "boundary_extract",
    "for_each_object", "for_each_object_translate", "for_each_object_recolor",
    "for_each_object_rotate", "for_each_object_scale", "for_each_object_flip"
]

DSL_OP_TO_IDX = {op: i for i, op in enumerate(DSL_OPS)}
IDX_TO_DSL_OP = {i: op for i, op in enumerate(DSL_OPS)}

# Total number of operations
NUM_DSL_OPS = len(DSL_OPS)

def get_op_index(op_name: str) -> int:
    """Get index for a DSL operation name"""
    return DSL_OP_TO_IDX.get(op_name, -1)

def get_op_name(op_index: int) -> str:
    """Get DSL operation name for an index"""
    return IDX_TO_DSL_OP.get(op_index, "unknown")

def is_valid_op(op_name: str) -> bool:
    """Check if operation name is valid"""
    return op_name in DSL_OP_TO_IDX

# Operation categories for better organization
GEOMETRIC_OPS = ["rotate90", "rotate180", "rotate270", "flip_h", "flip_v", "translate", "scale"]
COLOR_OPS = ["color_map", "extract_color", "for_each_object_recolor"]
SPATIAL_OPS = ["crop_bbox", "crop_nonzero", "tile", "paste", "resize_nn", "center_pad_to"]
LOGIC_OPS = ["grid_union", "grid_intersection", "grid_xor", "grid_difference"]
OBJECT_OPS = ["for_each_object", "for_each_object_translate", "for_each_object_recolor", 
             "for_each_object_rotate", "for_each_object_scale", "for_each_object_flip"]
ANALYSIS_OPS = ["count_objects", "count_colors", "find_pattern", "extract_pattern"]

OP_CATEGORIES = {
    "geometric": GEOMETRIC_OPS,
    "color": COLOR_OPS,
    "spatial": SPATIAL_OPS,
    "logic": LOGIC_OPS,
    "object": OBJECT_OPS,
    "analysis": ANALYSIS_OPS
}

def get_op_category(op_name: str) -> str:
    """Get category for a DSL operation"""
    for category, ops in OP_CATEGORIES.items():
        if op_name in ops:
            return category
    return "misc"