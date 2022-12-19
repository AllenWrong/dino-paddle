from reprod_log import ReprodDiffHelper


def check_diff(file1, file2, out_file, tresh=1e-6):
    diff_helper = ReprodDiffHelper()

    info1 = diff_helper.load_info(file1)
    info2 = diff_helper.load_info(file2)

    diff_helper.compare_info(info1, info2)

    diff_helper.report(
        diff_method="mean", diff_threshold=tresh, path=out_file)


def check_forward():
    check_diff("./pd/forward_pd.npy", "./th/forward_torch.npy", "./diff.log")


def check_forward_dino_loss():
    check_diff("./pd/forward_dino_loss_pd.npy", "./th/forward_dino_loss_th.npy", "./dino_loss_diff.log")


if __name__ == "__main__":
    check_forward()
    # check_forward_dino_loss()