import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_probabilities(df):
    total_yes = df['Play'].value_counts()['Yes']
    total_no = df['Play'].value_counts()['No']
    total = total_yes + total_no
    p_yes = total_yes / total
    p_no = total_no / total

    probabilities = {}
    features = ['Outlook', 'Temp', 'Humidity', 'Windy']
    for feature in features:
        probabilities[feature] = {}
        feature_values = df[feature].unique()
        for value in feature_values:
            yes_count = df[(df[feature] == value) & (df['Play'] == 'Yes')].shape[0]
            no_count = df[(df[feature] == value) & (df['Play'] == 'No')].shape[0]
            probabilities[feature][value] = {'Yes': yes_count / total_yes, 'No': no_count / total_no}

    return probabilities, p_yes, p_no

def predict(instance, probabilities, p_yes, p_no):
    prob_yes = p_yes
    prob_no = p_no

    for feature, value in instance.items():
        if value in probabilities[feature]:
            prob_yes *= probabilities[feature][value]['Yes']
            prob_no *= probabilities[feature][value]['No']

    total_prob = prob_yes + prob_no
    prob_yes /= total_prob
    prob_no /= total_prob

    return prob_yes, prob_no

def create_results_table(data, probabilities, p_yes, p_no):
    results = []
    for _, row in data.iterrows():
        instance = {
            'Outlook': row['Outlook'],
            'Temp': row['Temp'],
            'Humidity': row['Humidity'],
            'Windy': row['Windy']
        }
        prob_yes, prob_no = predict(instance, probabilities, p_yes, p_no)

        likelihood_yes = 1
        likelihood_no = 1
        log_likelihood_yes = 0
        log_likelihood_no = 0
        for feature, value in instance.items():
            if value in probabilities[feature]:
                likelihood_yes *= probabilities[feature][value]['Yes']
                likelihood_no *= probabilities[feature][value]['No']
                log_likelihood_yes += np.log(probabilities[feature][value]['Yes'])
                log_likelihood_no += np.log(probabilities[feature][value]['No'])

        evidence = likelihood_yes * p_yes + likelihood_no * p_no
        posterior_yes = likelihood_yes * p_yes / evidence
        posterior_no = likelihood_no * p_no / evidence

        result = {
            'Outlook': row['Outlook'],
            'Temp': row['Temp'],
            'Humidity': row['Humidity'],
            'Windy': row['Windy'],
            'Actual': row['Play'],
            'P(Yes)': prob_yes,
            'P(No)': prob_no,
            'Likelihood(Yes)': likelihood_yes,
            'Likelihood(No)': likelihood_no,
            'Log-Likelihood(Yes)': log_likelihood_yes,
            'Log-Likelihood(No)': log_likelihood_no,
            'Posterior(Yes)': posterior_yes,
            'Posterior(No)': posterior_no,
            'Prediction': 'Yes' if prob_yes > prob_no else 'No'
        }
        results.append(result)
    return pd.DataFrame(results)

def plot_results(results_df):
    plt.figure(figsize=(24, 16))

    ax1 = plt.subplot(3, 1, 1)
    sns.lineplot(data=results_df, x=results_df.index, y='P(Yes)', label='P(Yes)', color='blue', ax=ax1)
    sns.lineplot(data=results_df, x=results_df.index, y='P(No)', label='P(No)', color='orange', ax=ax1)
    sns.lineplot(data=results_df, x=results_df.index, y='Likelihood(Yes)', label='Likelihood(Yes)', color='green', ax=ax1)
    sns.lineplot(data=results_df, x=results_df.index, y='Likelihood(No)', label='Likelihood(No)', color='red', ax=ax1)

    ax1.set_title('Wave Plot of Probabilities and Likelihoods', fontsize=32, pad=20)
    ax1.set_xlabel('Instance Index', fontsize=24, labelpad=15)
    ax1.set_ylabel('Probability / Likelihood', fontsize=24, labelpad=15)
    ax1.tick_params(axis='both', labelsize=20)
    ax1.grid()
    ax1.legend(fontsize=20)

    outcome_counts = results_df['Actual'].value_counts()
    ax2 = plt.subplot(3, 1, 2)
    ax2.pie(outcome_counts, labels=outcome_counts.index, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'], textprops={'fontsize': 20})
    ax2.axis('equal')
    ax2.set_title('Distribution of Actual Outcomes', fontsize=32, pad=20)

    ax3 = plt.subplot(3, 1, 3)
    ax3.bar(results_df.index, results_df['P(Yes)'], label='P(Yes)', color='blue', alpha=0.6)
    ax3.bar(results_df.index, results_df['P(No)'], label='P(No)', color='orange', alpha=0.6)
    ax3.set_title('Bar Chart of Probabilities', fontsize=32, pad=20)
    ax3.set_xlabel('Instance Index', fontsize=24, labelpad=15)
    ax3.set_ylabel('Probability', fontsize=24, labelpad=15)
    ax3.tick_params(axis='both', labelsize=20)
    ax3.legend(fontsize=20)
    ax3.grid()

    plt.tight_layout()
    plt.show()

def main():
    data = pd.read_csv("play_data.csv")
    probabilities, p_yes, p_no = calculate_probabilities(data)
    results_df = create_results_table(data, probabilities, p_yes, p_no)

    print("\nResults Table:")
    print(results_df.to_string(index=False))

    correct_predictions = results_df['Prediction'] == results_df['Actual']
    accuracy = correct_predictions.mean()
    print(f"\nAccuracy: {accuracy:.2%}")

    plot_results(results_df)

if __name__ == "__main__":
    main()
