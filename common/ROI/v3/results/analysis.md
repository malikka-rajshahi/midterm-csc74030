# ROI V3 development diagnosis

Default: 42.5% GT coverage >=95%, 42.5% strict containment, 34.6 ROIs/image. Median/p90 compact area ratio at >=95%: 19.28/169.58. Merging reduces 498.5 unmerged ROIs/image but makes compact ROIs larger. The green/blue gate rejects 0 pre-merge candidates in this run; do not assume a filtering benefit. Lowering S to 0.25 yields 60.3% coverage with area-ratio p90 362.96: greater coverage can come from oversized boxes. These settings are provisional; the small, imbalanced training sample cannot establish an optimum.
