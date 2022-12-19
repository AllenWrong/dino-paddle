from reprod_log import ReprodDiffHelper


def check_diff(file1, file2, out_file, tresh=1e-6):
    diff_helper = ReprodDiffHelper()

    info1 = diff_helper.load_info(file1)
    info2 = diff_helper.load_info(file2)

    diff_helper.compare_info(info1, info2)

    diff_helper.report(
        diff_method="mean", diff_threshold=tresh, path=out_file)


def check_backward():
    check_diff("./pd_backward.npy", "./th_backward.npy", "backward_diff.log")


def check_lr():
    check_diff("./lr_pd.npy", "./lr_th.npy", "lr_diff.log")


def check_bkb_load():
    check_diff("./pd_bkb_state.npy", "./th_bkb_state.npy", "bkb_diff.log", 1e-9)


def check_linear_backward():
    check_diff("./clf_log_pd.npy", "./clf_log_th.npy", "clf_diff.log")
    check_diff("./clf_lr_pd.npy", "./clf_lr_th.npy", "clf_lr_diff.log")


def check_sub_m():
    check_diff("./sub_m_pd.npy", "./sub_m_th.npy", "sub_m_diff.log")


if __name__ == "__main__":
    # check_bkb_load()
    check_backward()
    # check_lr()
    # check_linear_backward()
    # check_sub_m()
