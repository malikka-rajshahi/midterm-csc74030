# ROI V2 development diagnosis

Default V2 fully contains 32/73 supplied GT boxes (43.8%), with 71.25 ROIs/image and approximately 0.31 seconds/image locally. Median/p90 compact ROI-to-GT area ratio at >=95% coverage is 3.32/15.01. Only 20/73 GTs satisfy >=95% coverage with area ratio <=4. This is not a solved proposal generator.

Merging cuts 498.5 unmerged candidates/image to 71.25 (85.7% reduction), increases >=95% coverage from 31 to 32 GTs, and increases strict containment from 30 to 32. However, median compact area ratio rises from 1.60 to 3.32, and IoU>=0.5 matches decrease from 31 to 13. This is the expected tradeoff of keeping larger enclosing regions, not a blanket improvement.

Both Green>40% and Green-or-Blue>50% reject zero candidates in this V2 sample, and their merged outputs equal the no-background-filter outputs. Earlier V1 filtering gains cannot be carried over to this different red/orange generator. This sample gives no evidence of benefit or damage from the V2 background gates.

Lowering the saturation threshold from 0.35 to 0.25 increases >=95% coverage to 44/73 and strict containment to 43/73, with 88.33 ROIs/image. But compact area-ratio p90 rises to 281.12, and only 21/73 GTs meet >=95% coverage with area ratio <=4. Much of the extra containment lacks compactness; the default remains 0.35 pending a larger reviewed training study. All chromatic evidence bands use this saturation threshold, not just red/orange.

Default class breakdown: cone 7/7, roadworks 0/3, vertical_pannel 25/63 strictly contained. Of 41 GTs missed at >=95%, 35 have fewer than four red/orange support pixels inside their supplied rectangle at working resolution; six have support but no containing proposal. This suggests inspecting weak-color/tiny targets and window/context placement first. The support count is observational, not proof of the failure cause. Broad roadworks group annotations also need visual review; the three-box count cannot support a class-level conclusion.

The bank was fitted on all official training GT, and evaluation uses a 12-image training subset. Labels remain unverified and imbalanced. No test images or classifiers were used. No global optimality or held-out performance is claimed.
