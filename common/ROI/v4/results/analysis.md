# ROI V4 development diagnosis

Default: 49.3% GT coverage >=95%, 49.3% strict containment, 6.8 ROIs/image. Median/p90 compact area ratio at >=95%: 259.68/683.75. Merging reduces 498.5 unmerged ROIs/image but makes compact ROIs larger. The green/blue gate rejects 0 pre-merge candidates in this run; do not assume a filtering benefit. Partial GT overlaps: 2; no overlap: 35. Oversized grouped ROIs/image: 0.00. Grouping removes intersecting outputs but may join multiple objects and much background. These settings are provisional; the small, imbalanced training sample cannot establish an optimum. Tree-only: rejected 12 anchors; 49.3% GT >=95%, 7.0 ROIs/image. Dark-only: 50.7% GT >=95%, 5.0 ROIs/image; extra overlap/coverage is not proof of correct background rejection or improved night detection.


The default has 36/73 complete GT coverage (>=95%), two partial GT overlaps and 35 GTs with no overlap. Strict containment is also 36/73. Partial overlap is not a verified detection. Grouped outputs have no positive-area overlaps on all 12 training images; the unmerged panel deliberately remains an overlapping diagnostic control.

The median eligible area ratio is 259.68 and only 3/73 GTs satisfy >=95% coverage with area ratio <=4. This demonstrates excessive background despite cleaner-looking grouped output. Absolute oversized flags (>25% image area) do not capture every box that is enormous relative to tiny GT.

The optional tree-context heuristic rejects 12 anchors, loses/gains zero covered GTs, and changes mean ROI count from 6.83 to 7.00 because deleting connecting support can split groups. There is no vegetation/false-positive ground truth to certify the rejected anchors as trees. The tree problem is not proven solved.

Dark-only extraction gains one GT overall, but no additional dark GT rectangles: 24/46 are covered with and without the option. These brightness groups use mean GT V<0.35, not verified day/night labels. Dark-subset median area ratio worsens from 257.99 to 6773.65. The option is not enabled by default and has not demonstrated better dark-target detection.

V1/V2/V3 remain independent. Official test data and classifiers are untouched. See README for rules, complete limitations and further work.
