"""
Week 8 Assignment - Machine Learning

Bryan Phillips
10/05/23
DATA 300/6381

Dataset: Churn.csv
"""

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set pandas display options
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.max_rows', None)  # Display all rows
pd.set_option('display.width', None)  # Width of the display in characters
pd.set_option('display.max_colwidth', None)  # Display the full content of each column


# pd.set_option('display.float_format', '{:.2f}'.format)  # Change exponential number display


def display_heatmap(features_scaled_dataset):
    """
    Plot heatmap encompassing all data

    :param features_scaled_dataset:
    :return: None (Displays the plot)
    """
    plt.figure(figsize=(20, 12))
    sns.heatmap(features_scaled_dataset.corr(), annot=True, cmap='RdYlGn')
    plt.title('Correlation Heatmap using Encoded and Scaled Data')
    plt.show()


def display_stackedbar_cservicecalls(dataset, dataset_v2):
    """
    Plot a stacked bar chart showing the distribution of customer service calls by churn status.

    :param dataset:
    :param dataset_v2:
    :return: None (Displays the plot)
    """
    # Use the original 'Customer service calls' data instead of the scaled data
    dataset_v2['Customer service calls'] = dataset['Customer service calls']

    # Set data in 'Customer service calls' column to integer for proper display
    dataset_v2['Customer service calls'] = dataset_v2['Customer service calls'].astype(int)

    # Stack the data for stacked bar graph type visualization
    stacked_data = dataset_v2.groupby(['Customer service calls', 'Churn_encoded']).size().unstack()

    # Plot the stacked bar chart
    ax = stacked_data.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title('Distribution of Customer Service Calls by Churn Status')
    plt.xlabel('Number of Customer Service Calls')
    plt.ylabel('Number of Customers')
    plt.legend(title='Churn', labels=['Did Not Churn', 'Churned'])

    # Set x-axis ticks to integer values directly from grouped index
    ax.set_xticklabels(stacked_data.index)

    plt.tight_layout()
    plt.show()


def display_boxplot_totaldayminutes(dataset_v2):
    """
    Plot a box plot comparing Total day minutes for Churned vs. Retained customers

    :param dataset_v2:
    :return: None (Displays the plot)
    """

    # Initialize the figure
    plt.figure(figsize=(10, 6))

    # Create the boxplot
    sns.boxplot(x='Churn_encoded', y='Total day minutes', data=dataset_v2)

    # Setting plot title and labels
    plt.title('Box plot of Total day minutes for Churned vs. Retained customers')
    plt.xlabel('Churn Status')
    plt.ylabel('Total day minutes')
    plt.xticks([0, 1], ['Retained', 'Churned'])  # Set the x-tick labels

    # Display the plot
    plt.show()


def display_scatter_timevscharge(dataset):
    """
    Plot a scatter plot that shows a correlation between day time service usage and service
    charge with the churn
    status representing each datapoint on the plot

    :param dataset:
    :return: None (Displays the plot)
    """

    # Create the scatter plot with hue for Churn status
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=dataset, x='Total day minutes', y='Total day charge', hue='Churn',
                    palette="coolwarm")

    # Create title for the scatter plot
    plt.title('Daytime Charges vs. Total Day Minutes by Churn Status')

    # Display the plot
    plt.show()


def main():
    """
    This section of the analysis focuses on preparing and cleaning the dataset.
    It provides basic statistics on key features of the dataset.
    """

    # Importing the dataset
    dataset = pd.read_csv('Churn.csv')

    # Drop the 'Area code' data because it is a geograph label rather than indicative of potential
    # customer churn.
    dataset = dataset.drop(['Area code'], axis=1)

    # Display the shape of the dataset
    print("Shape of the dataset:", dataset.shape)

    # Display the first five rows of the dataset
    print("\n")
    print(dataset.head(5))

    # Let’s now take a look at the number of instances (rows) that belong to
    # each class (Churn True/False).
    print("\n")
    print(dataset.groupby('Churn').size())

    # Determine which columns are categorical and which are numerical
    categorical = dataset.select_dtypes(include=[object])
    print("\n")
    print("Categorical Columns:", categorical.shape[1])

    numerical = dataset.select_dtypes(exclude=[object])
    print("\n")
    print("Numerical Columns:", numerical.shape[1])

    # Check for missing values
    print("\n")
    print("Missing values in dataset:", dataset.isnull().any().any())

    # ----------------------------------------------------------------------------------------------
    # Display basic statistics for all features
    print("\n")
    print(dataset.describe())

    # Average Customer Service Calls
    print("\n")
    print("The Average Customer Service Calls is : ",
          round(dataset['Customer service calls'].mean(), 2))

    # Average Customer Service Calls: Reflects overall customer satisfaction. Higher values might
    # suggest frequent issues, a potential factor for churn. Low averages may indicate higher
    # satisfaction.

    # Average International Call Charge
    print("The Average International Call Charge is : $",
          round(dataset['Total intl charge'].mean(), 2))

    # Average International Call Charge: Gauges a customer's typical international expense.
    # High charges might prompt customers to seek affordable alternatives, risking churn.

    # Average International Call Minutes
    print("The Average International Call Minutes is: ",
          round(dataset['Total intl minutes'].mean(), 2))

    # Average International Call Minutes: Indicates frequency of international calls. A high
    # average
    # suggests significant international communication reliance.

    # Average Total for Minutes per Daytime Hours
    print("The Average Total for Minutes per Daytime Hours is : ",
          round(dataset['Total day minutes'].mean(), 2))

    # Average total for Minutes per Daytime Hours: Provides insights into daytime usage patterns.

    # Average Daytime Domestic Call Charge
    print("The Average Daytime Domestic Call Charge is : $",
          round(dataset['Total day charge'].mean(), 2))

    # Average Daytime Domestic Call Charge: Helps in understanding daily customer expenses.
    # High rates might lead customers to seek better value elsewhere, risking churn.

    # Average Total for Minutes per Evening
    print("The Average Total for Minutes per Evening Time Hours is : ",
          round(dataset['Total eve minutes'].mean(), 2))

    # The Average Total for Minutes per Evening: Reflects customer behavior post-business hours.
    # High usage might indicate dependency during leisure time.

    # Average Evening Time Domestic Call Charge
    print("The Average Evening Time Domestic Call Charge is : $",
          round(dataset['Total eve charge'].mean(), 2))

    # Average Evening Time Domestic Call Charge: Evaluates perceived evening call value. High
    # charges
    # without perceived value may lead to customer dissatisfaction.

    # Scale & Encode Labels and Features for Heatmap visualization of all data and for later
    # KNN analysis
    # ----------------------------------------------------------------------------------------------

    # Features without target column 'Churn'
    features = dataset.drop('Churn', axis=1)
    target = dataset['Churn']

    # Encode the 'Churn' column
    label_encoder = LabelEncoder()
    target_encoded = label_encoder.fit_transform(target)

    # Scale the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Convert the scaled features back to a DataFrame for better readability
    features_scaled_dataset = pd.DataFrame(features_scaled, columns=features.columns)

    # Add the encoded 'Churn' column to the scaled features dataframe
    features_scaled_dataset['Churn_encoded'] = target_encoded

    # Display the first five rows of the dataset after encoding and scaling
    print("\nAfter Encoding and Scaling:\n")
    print(features_scaled_dataset.head(5))

    # ----------------------------------------------------------------------------------------------
    # Display Heatmap of all data after encoding and scaling
    """     
    The correlations between the various variables are shown visually by displaying a heatmap of 
    all 
    the X-features (columns) and Y-labels (Churn True/False column). The link between the features 
    and the target variable must be understood through this visualization to determine which 
    characteristics significantly impact customer attrition. After examining the heatmap, it is 
    possible to see:
    
        * Which characteristics indicate whether a client will continue their telecom services or 
        churn (cancel their service). 
        * Any issues with features that may be highly correlated and can be deemed redundant.  
        * After studying the heatmap, one can tailor the dataset to fit the KNN analysis better. 
        
    After looking at the heatmap, the features that stand out are:
        
        * 'Total day minutes'
        * 'Total day charge' 
        * 'Customer service calls' 
    
    They positively associate with the 'Churn' column, indicating that they are essential in 
    influencing whether suggesting they play a crucial role in determining whether a customer 
    churns 
    (cancels their services) or retains their subscription.
    """

    # Call the function to display the heatmap
    display_heatmap(features_scaled_dataset)

    # ----------------------------------------------------------------------------------------------
    # Create refined datasets based on information obtained from heatmap visualization and for later
    # use for KNN analysis

    # Extract the desired columns
    selected_features = ['Total day minutes', 'Total day charge', 'Customer service calls']
    dataset_v2 = features_scaled_dataset[selected_features].copy()
    dataset_v2['Churn_encoded'] = target_encoded

    # Extracting only 'Customer service calls' column for KNN analysis
    dataset_v3 = pd.DataFrame()
    dataset_v3['Customer service calls'] = features_scaled_dataset['Customer service calls'].copy()
    dataset_v3['Churn_encoded'] = target_encoded

    # Extracting 'Total day minutes' & 'Total day charge' column for KNN analysis
    dataset_v4_features = ['Total day minutes', 'Total day charge']
    dataset_v4 = features_scaled_dataset[dataset_v4_features].copy()
    dataset_v4['Churn_encoded'] = target_encoded

    # ----------------------------------------------------------------------------------------------
    # Visualization #1 Stacked bar chart - Customer Service Calls Distribution
    """
    The stacked bar chart shows the distribution of customer service calls and how they relate to 
    turnover, which is quite evident. This graph clarifies how the churn rate rises as the quantity 
    of customer service calls increases. This visualization offers insightful information, like how 
    even though fewer customers reached out to customer services between 4 and 6 times, 
    out of those 
    who called six times, those customers all churned and canceled their service. This insight 
    might 
    indicate that the more times a customer calls the service department, the more likely 
    they are to cancel their service.  
    """

    # Call the function to display the stacked bar graph
    display_stackedbar_cservicecalls(dataset, dataset_v2)

    # ----------------------------------------------------------------------------------------------
    # Visualization #2 Box plot - Distribution of Total day minutes for churned vs. retained
    # customers
    """
    The box plot illustrates how much time is spent on calls by the retained (customers who stayed) 
    and churned (customers who departed) groups during the day. "Total day minutes" denotes the 
    total minutes spent on each customer's calls.
    
    * Boxes: The boxes are the major components of the visualization. The 25th and 75th 
    percentiles are displayed on each box's top and bottom margins, respectively. Essentially, 
    these 
    boxes contain the call durations for half of each group.
    
        - Given the box's location at the bottom of the scale, calls to the "Retained" group are 
        often shorter.
        - Call durations for the "Churned" category are usually longer, as seen by the box's higher 
        position.
        
    * Line inside the Boxes (Median): A line inside each box, or the median, indicates the value 
    that falls in the middle. The numbers are split evenly between those above and below this line.
    
    * Whiskers: The lines that extend vertically from the boxes represent the range of call 
    durations outside the middle 50%. They aid in comprehending the call durations' overall 
    dispersion. There is a broader range of call durations if they are lengthy.
     
    * Dots: These are outliers. When compared to the others, the length of these calls is 
    particularly long or short.
    
    From this box plot, one can observe that the customers who churned (left the company) spent 
    more 
    time on calls during the day than those who stayed. This association might imply various 
    things, 
    like perhaps the churned customers who, from the visualization, used the service more during 
    the 
    day were not satisfied with the service and that it may have something to do with the cost of 
    the service during the day, perhaps they did not find value in the amount charged compared to 
    quality of service. In simple terms, Customers who left the company seem to spend more time on 
    the phone during the day than those who stayed. This correlation could indicate the are 
    unsatisfied with the service.    
    """

    # Call the function to display the box plot
    display_boxplot_totaldayminutes(dataset_v2)

    # ----------------------------------------------------------------------------------------------
    # Visualization #3 - Scatterplot
    """
    The scatterplot shows one how there is a distinct correlation between the customer's time spent 
    using the service during the day and how much they are being charged for their service. 
    Additionally, the scatterplot is color coded with the customer's churn status which is blue 
    for retained customers and orange for churned customers. 
    
        * Upward Trend: You'll notice almost a straight diagonal line going from the bottom left to 
        the top right. This association tells one that there's a clear relationship between the 
        amount of time customers use the service and how much they're charged.
        
        * Churn Status: At the right side of the plot, where the dots are further to the right 
        (indicating higher daytime use), one will notice more datapoints. This association suggests 
        that many customers who used the service alot during the day ended up leaving. One possible 
        reason could be that they found the service expensive for their usage levels.
        
        * High Cluster Area: Most customers seem clustered around the 100-200 minute mark and are 
        charged between $10-$30.This aspect of the scatterplot gives one an idea of the typical 
        usage and charge for most of our customer base.
    """

    # Call the function to display the scatter plot
    display_scatter_timevscharge(dataset)

    # ----------------------------------------------------------------------------------------------
    # Splitting the dataset into the Training set and Test set

    x_train, x_test, y_train, y_test = \
        (train_test_split(dataset_v4.drop('Churn_encoded', axis=1),
                          dataset_v4['Churn_encoded'],
                          test_size=0.2,  # using 80% for training and 20% for testing
                          random_state=42))  # Ensures consistency while testing

    # ----------------------------------------------------------------------------------------------
    # Fitting classifier to the Training set

    # Initiate the learning model (k = 3)
    classifier = KNeighborsClassifier(n_neighbors=3)

    # Fitting the model
    classifier.fit(x_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(x_test)

    # Evaluating predictions using a confusion matrix - This table essentially shows one how many
    # predictions were correct and many were not
    print("\n")
    print(confusion_matrix(y_test, y_pred))

    # Display confusion matrix in a more readable format using pandas
    print("\n")
    print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

    # Display a classification report for additional metrics
    print("\n")
    print(classification_report(y_test, y_pred))

    # Calculating model accuracy - Shows how often the model was correct
    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f'Accuracy of our model is: {round(accuracy, 2)} %.')

    # Using cross-validation for parameter tuning to see which number of neighbors is correct

    # Creating a list of odd numbers from 1 to 49 K for KNN
    k_list = list(range(1, 50, 2))
    cv_scores = []

    # Trying each number in the list as the number of neighbors
    for k in k_list:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())

    # Calculate the error rate for each number of neighbors
    mse = [1 - x for x in cv_scores]

    # Visualizing how the error rate changes as we use different numbers of neighbors
    plt.figure(figsize=(15, 10))
    plt.title('The optimal number of neighbors', fontsize=20, fontweight='bold')
    plt.xlabel('Number of Neighbors K', fontsize=15)
    plt.ylabel('Misclassification Error', fontsize=15)
    plt.plot(k_list, mse)
    plt.show()

    # Finding and displaying the number of neighbors that gave the lowest error rate
    best_k = k_list[mse.index(min(mse))]
    print(f"The optimal number of neighbors is {best_k}.")

    # Second Method
    # ----------------------------------------------------------------------------------------------
    # List to store how many times our model predicts a wrong value
    error_rate = []

    # Same as the first method, creating a list of possible neighbors from 1 to 49.
    for i in range(1, 50):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_train, y_train)
        pred = knn.predict(x_test)
        error_rate.append(np.mean(pred != y_test))

    # This plot will show how the error changes for different number of neighbors.
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 50), error_rate, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    plt.show()

    # GridSearch Cross-Validation - GridSearch is a tool for finding the best value for KNN
    # ----------------------------------------------------------------------------------------------
    # Specify a range of numbers between 1 and 50 to test
    param_grid = {'n_neighbors': np.arange(1, 50)}
    knn_cv = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    knn_cv.fit(features_scaled, target_encoded)

    # Show the accuracy score of our model and the best number for KNN.
    print(f"Best score during GridSearchCV: {knn_cv.best_score_}")
    print(f"Optimal number of neighbors: {knn_cv.best_params_['n_neighbors']}")


if __name__ == "__main__":
    main()
