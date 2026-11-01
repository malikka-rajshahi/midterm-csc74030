# Geometry refinement development findings

FINAL keeps 19 S2 shapes and adds 3 scale-boundary S1 shapes: [(128, 512), (256, 512), (512, 512)]. Only global upper size bounds are rounded to the power-of-two grid; lower and aspect limits stay unchanged. No perspective gate or image-specific GT controls generation. This is a final candidate-generation configuration, not object recognition; feature matching remains pending. Overlapping local alternatives are intentional. GT-best images are oracle diagnostics, not runtime selections.
