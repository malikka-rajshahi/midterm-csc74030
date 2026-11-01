"""Run the latest V4 evaluator; historical versions have independent entry points."""
if __name__ == '__main__':
    import runpy
    runpy.run_module('common.ROI.v4.evaluate_rois', run_name='__main__')
else:
    from .v4.evaluate_rois import evaluate, fit_config, measure_gt, read_training, select_images
