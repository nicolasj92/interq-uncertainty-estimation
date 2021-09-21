import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def get_correct_and_probs_sorted(model, X_test, y_test):
    y_pred = model.predict(X_test)
    probs = np.amax(model.predict_proba(X_test), axis=1)
    correct = np.array(y_pred == y_test)

    inds = np.argsort(probs)
    sorted_probs = probs[inds]
    sorted_correct = correct[inds]
    return sorted_correct, sorted_probs


def plot_accuracy_over_confidence_thresholds(models, X, y, plot_fraction_of_samples=False, title="Accuracy by confidence thresholds"):
    plt.figure(figsize=(10, 10))
    color = iter(plt.cm.Dark2(np.linspace(0, 1, len(models))))
    thresholds = np.arange(0., 1.05, 0.09)
    for name, model_runs in models.items():
        c = next(color)
        fractions = np.zeros((len(model_runs), len(thresholds)))
        accuracies = np.zeros((len(model_runs), len(thresholds)))
        for i, model in enumerate(model_runs):
            correct, probs = get_correct_and_probs_sorted(model, X, y)
            for j, thresh in enumerate(thresholds):
                mask = probs >= thresh
                accuracies[i, j] = np.sum(correct[mask]) / len(correct[mask])
                fractions[i, j] = np.sum(mask) / len(correct)

        mean = np.mean(accuracies, axis=0)
        std = np.std(accuracies, axis=0)
        plt.plot(thresholds, mean, "s--", c=c,
                 label="Accuracy ({})".format(name))
        plt.fill_between(thresholds, mean-std, mean+std, color=c, alpha=0.2)
        if plot_fraction_of_samples:
            mean_fractions = np.mean(fractions, axis=0)
            std_fractions = np.std(fractions, axis=0)
            plt.plot(thresholds, mean_fractions, "x--", c=c,
                     label="Fraction of samples ({})".format(name))
            plt.fill_between(thresholds, mean_fractions-std_fractions,
                             mean_fractions + std_fractions, color=c, alpha=0.2)

    plt.xlim(-0.05, 1.0)
    plt.ylim(-0.05, 1.05)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


def calibration_curve_and_error(corrects, confs, n_bins=10):
    mean_accuracies = []
    mean_confidences = []
    fraction_of_samples = []
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    conf_bin_inds = np.digitize(confs, bin_edges, right=True) - 1
    for i in range(n_bins):
        relevant_corrects = corrects[conf_bin_inds == i]
        relevant_confs = confs[conf_bin_inds == i]

        if len(relevant_corrects) > 0:
            mean_accuracies.append(
                np.sum(relevant_corrects) / len(relevant_corrects))
            mean_confidences.append(np.mean(relevant_confs))
            fraction_of_samples.append(len(relevant_corrects) / len(corrects))

    mean_accuracies = np.array(mean_accuracies)
    mean_confidences = np.array(mean_confidences)
    fraction_of_samples = np.array(fraction_of_samples)

    expected_calibration_error = np.sum(
        fraction_of_samples*np.abs(mean_accuracies-mean_confidences))

    return mean_accuracies, mean_confidences, expected_calibration_error


def plot_calibration_curve(models, X, y, title="Confidence calibration"):
    plt.figure(figsize=(10, 10))
    color = iter(plt.cm.Dark2(np.linspace(0, 1, len(models))))
    for name, model_runs in models.items():
        c = next(color)
        eces = []
        for i, model in enumerate(model_runs):
            if isinstance(X, list) and isinstance(y, list):
                correct, probs = get_correct_and_probs_sorted(
                    model, X[i], y[i])
            else:
                correct, probs = get_correct_and_probs_sorted(model, X, y)
            frac_correct, mean_prob, ece = calibration_curve_and_error(
                correct, probs, n_bins=10)
            eces.append(ece)

        # Only plot last models curve as this cannot be easily averaged
        plt.plot(mean_prob, frac_correct, 's--', c=c,
                 label="{} - ECE Mean: {} Std: {}".format(name, np.round(np.mean(eces), 2), np.round(np.std(eces), 2)))

    plt.plot([0, 1], [0, 1], 'k:', label="Perfect Calibration")
    plt.xlabel("Average Confidence")
    plt.ylabel("Average Accuracy")
    plt.title(title)
    plt.xlim(-0.05, 1.05)
    plt.grid()
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.show()


def plot_over_batches(models, X_test, y_test, batch_start=0, batch_end=10, X_val=None, y_val=None, plot_accuracy=False, start_label=False, stop_label=False, plot_confidence=False, plot_entropy=False, title="", savename=None):
    plt.figure(figsize=(9, 5))
    color = iter(plt.cm.Dark2(np.linspace(0, 1, len(models))))
    add_val = X_val is not None and y_val is not None
    batch_ids = np.arange(batch_start, batch_end)

    for name, model_runs in models.items():
        c = next(color)
        if add_val:
            confidences = np.zeros((len(model_runs), 1+len(batch_ids)))
            entropies = np.zeros((len(model_runs), 1+len(batch_ids)))
            accuracies = np.zeros((len(model_runs), 1+len(batch_ids)))
        else:
            confidences = np.zeros((len(model_runs), len(batch_ids)))
            entropies = np.zeros((len(model_runs), len(batch_ids)))
            accuracies = np.zeros((len(model_runs), len(batch_ids)))

        for i, model in enumerate(model_runs):

            if add_val:
                X = X_val[i]
                y = y_val[i]

                if plot_confidence:
                    y_conf = model.predict_proba(X)
                    confidences[i, 0] = np.mean(np.amax(y_conf, axis=1))

                if plot_entropy:
                    y_entropy = model.predict_entropy(X)
                    entropies[i, 0] = y_entropy

                if plot_accuracy:
                    y_pred = model.predict(X)
                    accuracy = accuracy_score(y, y_pred)
                    accuracies[i, 0] = accuracy

            for j, batch_id in enumerate(batch_ids):
                mask = X_test["batch"] == batch_id
                X = X_test[mask].drop("batch", axis=1)
                y = y_test[mask]

                if plot_confidence:
                    y_conf = model.predict_proba(X)
                    confidences[i, j + int(add_val)
                                ] = np.mean(np.amax(y_conf, axis=1))

                if plot_entropy:
                    y_entropy = model.predict_entropy(X)
                    entropies[i, j + int(add_val)] = y_entropy

                if plot_accuracy:
                    y_pred = model.predict(X)
                    accuracy = accuracy_score(y, y_pred)
                    accuracies[i, j + int(add_val)] = accuracy

        plot_batch_ids = batch_ids + 1
        if add_val:
            plot_batch_ids = np.insert(batch_ids, 0, np.min(batch_ids) - 1)

        if plot_accuracy:
            mean_accuracy = np.mean(accuracies, axis=0)
            std_accuracy = np.std(accuracies, axis=0)
            plt.plot(plot_batch_ids, mean_accuracy, 's--', c=c,
                     label="Accuracy ({})".format(name))
            plt.fill_between(plot_batch_ids, mean_accuracy-std_accuracy,
                             mean_accuracy + std_accuracy, color=c, alpha=0.2)

        if plot_confidence:
            mean_confidence = np.mean(confidences, axis=0)
            std_confidence = np.std(confidences, axis=0)
            plt.plot(plot_batch_ids, mean_confidence, 'x--', c=c,
                     label="Confidence ({})".format(name))
            plt.fill_between(plot_batch_ids, mean_confidence-std_confidence,
                             mean_confidence + std_confidence, color=c, alpha=0.2)

        if plot_entropy:
            mean_entropy = np.mean(entropies, axis=0)
            std_entropy = np.std(entropies, axis=0)
            plt.plot(plot_batch_ids, mean_entropy, '*--', c=c,
                     label="Entropy ({})".format(name))
            plt.fill_between(plot_batch_ids, mean_entropy-std_entropy,
                             mean_entropy + std_entropy, color=c, alpha=0.2)

    plt.xlabel("Time")
    plt.ylim(0.0, 1.0)

    if add_val:
        tick_locations = plot_batch_ids
        tick_text = [str(i + 2) for i in range(len(plot_batch_ids))]

        if start_label and stop_label:
            tick_text = ["--"] * len(plot_batch_ids)
            tick_text[1] = start_label
            tick_text[-1] = stop_label

        tick_text[0] = "Val"

        plt.xticks(ticks=tick_locations, labels=tick_text)

    plt.grid()
    plt.legend()
    plt.title(title)

    if savename is not None:
        plt.savefig(savename)

    plt.show()


def plot_accuracy_and_confidence(models, X_test, y_test, batch_start=0, batch_end=10, X_val=None, y_val=None, plot_std=False, start_label=False, stop_label=False, title="", savename=None, legend=False):
    plt.figure(figsize=(9, 5))
    color = iter(plt.cm.Dark2(np.linspace(0, 1, len(models))))
    add_val = X_val is not None and y_val is not None
    batch_ids = np.arange(batch_start, batch_end)

    fig, axs = plt.subplots(2)

    for name, model_runs in models.items():
        c = next(color)
        if add_val:
            confidences = np.zeros((len(model_runs), 1+len(batch_ids)))
            accuracies = np.zeros((len(model_runs), 1+len(batch_ids)))
        else:
            confidences = np.zeros((len(model_runs), len(batch_ids)))
            accuracies = np.zeros((len(model_runs), len(batch_ids)))

        for i, model in enumerate(model_runs):

            if add_val:
                X = X_val[i]
                y = y_val[i]

                y_conf = model.predict_proba(X)
                confidences[i, 0] = np.mean(np.amax(y_conf, axis=1))

                y_pred = model.predict(X)
                accuracy = accuracy_score(y, y_pred)
                accuracies[i, 0] = accuracy

            for j, batch_id in enumerate(batch_ids):
                mask = X_test["batch"] == batch_id
                X = X_test[mask].drop("batch", axis=1)
                y = y_test[mask]

                y_conf = model.predict_proba(X)
                confidences[i, j + int(add_val)
                            ] = np.mean(np.amax(y_conf, axis=1))

                y_pred = model.predict(X)
                accuracy = accuracy_score(y, y_pred)
                accuracies[i, j + int(add_val)] = accuracy

        plot_batch_ids = batch_ids + 1
        if add_val:
            plot_batch_ids = np.insert(batch_ids, 0, np.min(batch_ids) - 1)

        mean_accuracy = np.mean(accuracies, axis=0)
        std_accuracy = np.std(accuracies, axis=0)
        axs[0].plot(plot_batch_ids, mean_accuracy, 'x--', c=c,
                    label="Accuracy ({})".format(name))
        if plot_std:
            axs[0].fill_between(plot_batch_ids, mean_accuracy-std_accuracy,
                                mean_accuracy + std_accuracy, color=c, alpha=0.2)

        mean_confidence = np.mean(confidences, axis=0)
        std_confidence = np.std(confidences, axis=0)
        axs[1].plot(plot_batch_ids, mean_confidence, 'x--', c=c,
                    label="Confidence ({})".format(name))
        if plot_std:
            axs[1].fill_between(plot_batch_ids, mean_confidence-std_confidence,
                                mean_confidence + std_confidence, color=c, alpha=0.2)

    axs[1].set_xlabel("Time")
    axs[1].set_ylim(0.25, 1.0)
    axs[0].set_ylim(0.25, 1.0)

    axs[0].set_ylabel("Accuracy")
    axs[1].set_ylabel("Confidence")

    if add_val:
        tick_locations = plot_batch_ids
        tick_text = [str(i + 2) for i in range(len(plot_batch_ids))]

        if start_label and stop_label:
            tick_text = ["--"] * len(plot_batch_ids)
            tick_text[1] = start_label
            tick_text[-1] = stop_label

        tick_text[0] = "Val"

        axs[0].set_xticks([])
        axs[1].set_xticks(ticks=tick_locations)
        axs[1].set_xticklabels(labels=tick_text)

    axs[1].grid(axis="y")
    axs[0].grid()

    fig.subplots_adjust(hspace=0.05)

    if legend:
        axs[1].legend(loc="lower left", prop={'size': 12}, ncol=2)

    if savename is not None:
        plt.savefig(savename)

    plt.show()
