import os
from typing import (
    Any,
    Dict
)


def analyze_model(
    model,
    x_train,
    x_validation,
    x_bt,
    y_train,
    y_validation,
    y_bt,
    x_train_all,
    y_train_all,
    title='QSRR Model'
) -> Dict[str, Any]:

    # Predictions
    _y_train_hat = model.predict(x_train)
    _y_validation_hat = model.predict(x_validation)

    # predict
    y_blindtest = model.predict(X_test)

    # metrics
    # calculate R2 and RMSE for train and validation Data

    # train
    r2_model_train = r2_score(y_train, y_cal)
    rmse_model_train = mean_squared_error(y_train, y_cal, squared=False)
    # Mape_model_valid=mean_absolute_percentage_error(y_train, y_cal,)

    # valid
    r2_model_valid = r2_score(y_valid, y_pred)
    rmse_model_valid = mean_squared_error(y_valid, y_pred, squared=False)
    # Mape_model_valid=mean_absolute_percentage_error(y_valid,y_pred,squared=False)

    # blidn test performance
    rmse_model_test = mean_squared_error(y_test, y_blindtest, squared=False)
    r2_model_test = r2_score(y_test, y_blindtest)
    # Mape_model_valid=mean_absolute_percentage_error(y_test,y_blindtest,)

    # cross validation
    # # kf= KFold(n_splits=5, shuffle=True, random_state=6)
    # r2_cv= cross_val_score(model, X_train, y_train, scoring= 'r2',cv=cv)
    # rmse_cv= cross_val_score(model, X_train, y_train, scoring= 'neg_root_mean_squared_error' ,cv=cv)

    ###############################
    # results fstring
    print("\nModel Report")
    print("=" * 80)

    # train
    print(f'r2_train: {r2_model_train:.3f}')
    print(f"RMSE_train: {rmse_model_train:.3f} min")
    print("-" * 80)

    # valid
    print(f'r2_valid: {r2_model_valid:.3f}')
    print(f"RMSE_valid: {rmse_model_valid:.3f} min")
    print("-" * 80)

    # blindtest
    print(f'r2_test: {r2_model_test:.3f}')
    print(f"RMSE_test: {rmse_model_test:.3f} min")
    print("-" * 80)

    ###############################    Residual_plot    ####################################
    if residualplot:
        # Calculate the residuals for train data
        if reshape == True:
            # reshape : "there is  an error that force me to reshape our y data"
            y_train = y_train.reshape(-1, 1)  # from (n,) to (n,1)
            y_test = y_test.reshape(-1, 1)
            y_valid = y_valid.reshape(-1, 1)

        residuals = (y_train - y_cal)
        residuals_test = (y_test - y_blindtest)
        residuals_valid = (y_valid - y_pred)

        # Create subplots with adjusted widths
        fig, (ax_res, ax_hist) = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={'width_ratios': [3, 1]})

        # Residual plot for train data
        ax_res.axhline(y=0, color='black', alpha=0.8, linewidth=0.7, label='_nolegend_')
        ax_res.scatter(y_cal, residuals, color=train_color, marker='o', alpha=0.8, edgecolors='black', linewidths=0.5)
        ax_res.scatter(y_pred, residuals_valid, marker='^', color=valid_color, alpha=0.8, edgecolors='black',
                       linewidths=0.5)
        ax_res.scatter(y_blindtest, residuals_test, color=test_color, marker='s', alpha=0.5, edgecolors='black',
                       linewidths=0.5)
        ax_res.set_xlabel('Predicted Values')
        ax_res.set_ylabel('Residuals')
        ax_res.set_title('Residual Plot')

        # Histogram plot for train data
        ax_hist.axhline(y=0, color='black', alpha=0.5, linewidth=0.6)
        ax_hist.hist(residuals, bins=50, color=train_color, alpha=0.8, orientation='horizontal', edgecolor='black')
        ax_hist.hist(residuals_test, bins=50, color=test_color, alpha=0.6, orientation='horizontal', edgecolor='black')
        ax_hist.hist(residuals_valid, bins=50, color=valid_color, alpha=0.5, orientation='horizontal',
                     edgecolor='black')

        legend = ax_res.legend(['Train', 'Validation', 'Test'], loc="best", ncol=3, fontsize=10)
        ax_hist.set_xlabel('Distribution')
        ax_hist.set_title('Residual Histogram')
        ax_hist.yaxis.tick_right()

        # Adjust spacing between subplots
        plt.tight_layout()

        # Display the plots
        plt.grid(False)
        plt.savefig(f'{out_path}/Residual_plot.svg', dpi=300, bbox_inches='tight', format='svg')
        plt.show()

    #################################    William Plot      ###########################
    if william_plot:
        # Hat matrix and Leverage
        # formula: X . inv(XT.X) . X.T
        print("[INFO] Apllicability Domain with william plot Evaluation...")
        # data x, valid
        X_new = np.concatenate((X_train, X_valid, X_test))
        xtx = X_new.T.dot(X_new)
        rand = np.random.uniform(1e-6, 1e-7, (len(np.diag(xtx)),))  # avoid singular matrix
        a = np.diag(xtx) + rand
        np.fill_diagonal(xtx, a)  # fill diagonal with new diagonal values
        hat = X_new.dot(np.linalg.inv(xtx)).dot(X_new.T)  # hat matrix
        hat_diag = np.diagonal(hat)  # diagonal values = leverage
        trace = hat_diag.sum()

        #### Standard residual   train set

        if reshape == True:
            y_train = y_train.reshape(-1)  # from (n,1) to (n)
            y_test = y_test.reshape(-1)
            y_valid = y_valid.reshape(-1)

        standard_scalar = StandardScaler()

        residual = y_train - y_cal.ravel()
        residual1 = y_valid - y_pred.ravel()
        residual2 = y_test - y_blindtest.ravel()

        residual_train_valid = np.concatenate((residual, residual1, residual2))
        sc = standard_scalar.fit(residual_train_valid.reshape(-1, 1))

        residual = sc.transform(residual.reshape(-1, 1))
        residual1 = sc.transform(residual1.reshape(-1, 1))
        residual2 = sc.transform(residual2.reshape(-1, 1))

        # join
        standard_residual = np.concatenate((residual, residual1, residual2))

        h_star = (3 * (X_new.shape[1] + 1)) / (len(X_train) + len(X_valid))  # h_star -> (h*)= 3p/n p=15+1   n=252

        ###### plot : william plot
        #  scatter plot of results
        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(hspace=0.5, wspace=0.7)

        ax1 = plt.subplot()
        # plot train
        ax1.scatter(hat_diag[0:X_train.shape[0]], standard_residual[0:X_train.shape[0]], c=train_color, marker="o",
                    alpha=0.8, edgecolors='black', linewidths=0.5)
        # # plot valid
        ax1.scatter(hat_diag[X_train.shape[0]:len(X_train) + len(X_valid)],
                    standard_residual[X_train.shape[0]:len(X_train) + len(X_valid)], c=valid_color, marker="^",
                    alpha=0.5, edgecolors='black', linewidths=0.6)
        # # plot valid
        ax1.scatter(hat_diag[len(X_train) + len(X_valid):], standard_residual[len(X_train) + len(X_valid):],
                    facecolor=test_color, marker="s", alpha=0.4, edgecolors='black', linewidths=0.5)

        # draw warning limits
        ax1.axhline(3, ls=":", color="tomato", lw=1.5)
        ax1.axhline(-3, ls=":", color="tomato", lw=1.5)
        ax1.axvline(h_star, ls=":", color="tomato", lw=1.5)

        h_s = f'h* = {h_star:.2f}'  # h*
        ax1.text(h_star - 0.025, -2.5, h_s, ha='left', va='top', fontsize=12)

        for i in np.arange(0, 252):
            if standard_residual[i] >= 3 or standard_residual[i] <= -3 or hat_diag[i] >= h_star:
                ax1.annotate(f'{i}', (hat_diag[i], standard_residual[i]), fontsize=12)

        # legend
        ax1.legend(('Train', 'Validation', 'Test'), loc=(0.7, 0.8))
        # limit -6 to 6 in y axis each  1 ticks
        ax1.set_ylim(-6, 6)
        ax1.yaxis.set_ticks(np.arange(-6, 7, 1))

        # x and y label
        ax1.set_xlabel("Leverage")
        ax1.set_ylabel("Standardized Residuals")
        # ax1.set_title(f"\n{title} model") #tR Experiment - tR Predicted
        plt.grid(False)
        plt.savefig(f'{out_path}/william_plot_{title}.svg', dpi=300, bbox_inches='tight', format='svg')
        plt.show()

    #################################    Y-randomization   ####################################
    if Y_Random:
        print("[INFO] Y-randomization Evaluation...")
        from sklearn.model_selection import permutation_test_score
        scores = ['neg_root_mean_squared_error']
        for score in scores:
            score0, perm_scores, pvalue = permutation_test_score(
                model, X_tot, y_tot, scoring=score, cv=ps, n_permutations=1000)
            print('Average score after 1000 iteration:', -perm_scores.mean())

            fig, ax = plt.subplots(figsize=(6, 6))

            ax = sns.histplot(data=-perm_scores, color=y_random_color, bins=50, edgecolor='black', alpha=0.7)

            ax.axvline(-score0, ls="--", color="r")
            score_label = f"RMSE on validation data: {-score0:.2f} min\n(p-value: {pvalue:.3f} )"

            ax.text(0.08, 0.7, score_label, ha='left', va='top', transform=ax.transAxes)

            #   ax.text(0.7, 0.5, score_label, fontsize=9)
            ax.set_xlabel("RMSE_val score")
            _ = ax.set_ylabel("Probability")
            # ax.set_title('Y-Randomization plot')
            plt.grid(False)
            plt.savefig(f'{out_path}/Y-Randomization.svg', dpi=300, bbox_inches='tight', format='svg')
            plt.show()
    #################################    Learning curve    ####################################
    if Learning_curve:
        from sklearn.model_selection import learning_curve
        print("[INFO] Learning curve....")
        ##learning curve
        # Use learning curve to get training and validation scores along with train sizes
        #
        train_sizes, train_scores, test_scores = learning_curve(estimator=model, X=X_tot, y=y_tot,
                                                                cv=10, train_sizes=np.linspace(0.1, 1.0, 10),
                                                                scoring='r2', n_jobs=1)
        #
        # Calculate training and test mean and std
        #
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        #
        # Plot the learning curve
        fig, ax = plt.subplots()
        plt.plot(train_sizes, train_mean, color=train_color, marker='o', markersize=5, label='Training Accuracy')
        plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')

        plt.plot(train_sizes, test_mean, color=valid_color, marker='+', markersize=5, linestyle='--',
                 label='Validation Accuracy')
        plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color=valid_color)
        plt.title('Learning Curve')
        plt.xlabel('Training Data Size')
        plt.ylabel('Model accuracy')
        # plt.grid()
        plt.legend(loc='lower right')
        plt.grid(False)
        plt.savefig(f'{out_path}/Learning_curve.svg', dpi=300, bbox_inches='tight', format='svg')
        plt.show()
