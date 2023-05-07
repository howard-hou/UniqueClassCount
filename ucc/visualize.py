import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools

m_color = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#9a6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#808080",
    "#ffffff",
    "#000000",
]


def visualize_distributions(distributions, classes, num_bins, fig_filename=None):
    hist_max = np.amax(distributions)
    num_features = int(distributions.shape[1] / num_bins)
    fig, ax = plt.subplots(len(classes), num_features, figsize=(16, 10))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.95, bottom=0.1)

    legend_color = list()
    for h in range(num_features):
        temp_hist_data = distributions[:, h * num_bins : (h + 1) * num_bins]

        for c in range(len(classes)):
            data = temp_hist_data[c, :]

            ax1 = ax[c, h]
            (bp,) = ax1.plot(np.arange(0, 1.1, 0.1), data, c=m_color[c], linestyle="-")

            if h == (num_features - 1):
                legend_color.append(bp)

            ax1.tick_params(axis="both", labelsize=6)
            ax1.tick_params(axis="x", rotation=90)
            ax1.set_ylim((-0.05, hist_max + 0.05))
            ax1.yaxis.set_ticks(np.arange(0, hist_max + 0.05, 0.1))
            ax1.yaxis.grid(
                True, linestyle="-", which="major", color="lightgrey", alpha=0.5
            )
            ax1.set_xlim((-0.1, 1.1))
            ax1.xaxis.set_ticks(np.arange(0, 1.1, 0.2))

            ax1.xaxis.grid(
                True, linestyle="-", which="major", color="lightgrey", alpha=0.5
            )
            ax1.set_axisbelow(True)

    fig.tight_layout()
    fig.subplots_adjust(left=0.03, right=0.98, top=0.96, bottom=0.10)
    fig.legend(
        legend_color,
        classes,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        fancybox=True,
        shadow=True,
        ncol=10,
        fontsize=15,
    )

    if fig_filename is not None:
        fig.savefig(fig_filename, bbox_inches="tight")
    plt.show()


def visualize_reconstucted_images(reconstructed_imgs, fig_filename=None):
    # generate a plot with 5 columns, rows depends on the number of images
    num_rows = len(reconstructed_imgs) // 5

    fig, ax = plt.subplots(num_rows, 5, figsize=(10, 5))
    for j in range(reconstructed_imgs.shape[0]):
        ax[j // 5, j % 5].imshow(reconstructed_imgs[j, 0, :, :])
        ax[j // 5, j % 5].axis("off")

    if fig_filename is not None:
        fig.savefig(fig_filename, bbox_inches="tight")

    plt.show()


def visualize_js_divergence(js_divergence_arr, classes):
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(
        js_divergence_arr, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1
    )
    ax.set_title("$D_{\mathcal{JS}}(\mathcal{P}||\mathcal{Q})$", fontsize=16)
    fig.colorbar(im)

    # Set font size of tick labels
    ax.tick_params(axis="both", which="major", labelsize=12)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=0)
    ax.set_yticklabels(classes)

    fmt = ".3f"
    for m, n in itertools.product(
        range(js_divergence_arr.shape[0]), range(js_divergence_arr.shape[1])
    ):
        ax.text(
            n,
            m,
            format(js_divergence_arr[m, n], fmt),
            horizontalalignment="center",
            color="black",
        )
    # set x, y label and set font size
    ax.set_ylabel("$\mathcal{P}$", fontsize=16)
    ax.set_xlabel("$\mathcal{Q}$", fontsize=16)
    plt.show()


def plot_confusion_matrix(cm, classes,
						  normalize=False,
						  title='Confusion matrix',
						  cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	# print(cm)

	plt.rcParams.update({'font.size':12, 'font.family':'sans-serif'})
	ax = plt.gca()
	im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	# plt.title(title)
	# plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=0)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",verticalalignment="center",
				 fontsize=16,
				 color="white" if cm[i, j] > thresh else "black")

	plt.ylabel('True Label')
	plt.xlabel('Predicted Label')
	
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)

	plt.colorbar(im, cax=cax)
	plt.tight_layout()
