
class MetricCalculator():
    def __init__(self, opt) -> None:
        self.grasp_succ_rate = None
        self.grasp_coverage_rate = None
        self.obj_succ_rate = None
        self.grasp_pred_succ_num = 0.
        self.grasp_pred_num = 0.
        self.grasp_gt_cov_num = 0.
        self.grasp_gt_num = 0.
        self.obj_succ_num = 0.
        self.obj_num = 0.

    def update(self, pred_num, gt_num, pred_succ_num, gt_cover_num):
        """Update the info for one object"""
        # parse results
        self.grasp_pred_num += pred_num
        self.grasp_pred_succ_num += pred_succ_num
        self.grasp_gt_num += gt_num
        self.grasp_gt_cov_num += gt_cover_num
        self.obj_num += 1
        if pred_succ_num >= 1:
            # if any grasp works, then the object is counted as being successfully grasped
            self.obj_succ_num += 1
    
    def get_metrics(self):
        # calculate grasp success rate
        self.grasp_succ_rate = self.grasp_pred_succ_num / self.grasp_pred_num if self.grasp_pred_num > 0 else -1
        # calculate grasp coverage rate
        self.grasp_coverage_rate = self.grasp_gt_cov_num / self.grasp_gt_num if self.grasp_gt_num > 0 else -1
        # calculate object success rate
        self.obj_succ_rate = self.obj_succ_num / self.obj_num if self.obj_num > 0 else -1
        return self.grasp_succ_rate, self.grasp_coverage_rate, self.obj_succ_rate
    
    def get_nums(self):
        return self.grasp_pred_succ_num, self.grasp_pred_num, self.grasp_gt_cov_num, self.grasp_gt_num, self.obj_succ_num, self.obj_num
    
    def print_nums(self):
        print("The pred_grasp_succ_num / pred_grasp_number / the GT_grasp_cov_num /"
         "GT_grasp_num / obj_succ_num /  obj_num are:")
        print(self.get_nums())
    
    def print_results(self, name, NA_gsr=False, NA_gcr=False, NA_osr=False):
        self.get_metrics()
        grasp_succ_rate = -1 if NA_gsr else self.grasp_succ_rate
        grasp_coverage_rate = -1 if NA_gcr else self.grasp_coverage_rate
        obj_succ_rate = -1 if NA_osr else self.obj_succ_rate
        print("The metrics for {:<40}: {:.4f} / {:.4f} / {:.4f}".format(name,
            grasp_succ_rate, grasp_coverage_rate, obj_succ_rate))
