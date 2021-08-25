# sentiment-analysis
Sentiment analysis of iPhone and Samsung Galaxy.

A full report is detailed in "Sentiment Analysis Report.pdf"

Overview:

Helio is a mobile app development company that is currently working with a U.S. government health agency to develop several smartphone apps for use by medical workers providing aid in developing countries. They would like a single phone model to be chosen for the sake of streamlining the training process and ease of technical support services. They would also like to choose a model that will be popular among the workers.
Alert! Analytic’s role in this project is to conduct a broad-based web sentiment analysis of two popular smartphone models -- the iPhone and Samsung Galaxy. Our goal is to understand the overall attitude towards each of these devices, so that Helio can choose their smartphone model with confidence.

Methods:

Data Collection-

Data was collected from the Common Crawl, an open repository of archived data from the web, consisting of billions of webpages to date. Using Amazon Web Services (AWS) and Elastic Map Reduce (EMR), we were able to build a large data matrix consisting of over 20,000 instances of pages that were relevant to our analysis. In order to assess whether a document was relevant, words indicating that the page was about the device and presented a meaningful analysis of the device were searched for. A document was only used if it contained at least one mention of the device and contained at least one of the following terms: review, critique, looks at, in depth, analysis, evaluate, evaluation, assess. 
Once relevant documents were found, information was collected about the sentiment towards features of each phone, including the display, camera, performance, and operating system. The script searched for and counted positive, negative, and neutral words that were within 5 words of a feature word (such as camera or display). 
Additionally, two small data matrices were created using this same method for the purpose of model training, in which the overall sentiment towards each phone was determined manually by our team. Using a trained model, the goal was to be able to predict overall sentiment towards each phone in the large data matrix that was collected.

Model Building and Evaluation:

Initially, four different models were tested: C5.0, Random Forest, SVM, and KKNN. They were tested on the out of box dataset, NZV dataset, and RFE dataset for both the iPhone and Samsung Galaxy.
 At first, the dependent variable consisted of 6 sentiment classes. This proved difficult for each of the models, as the 0 (negative) and 5 (positive) classes were the most abundant and therefore more easily learned, while the other 4 classes were very difficult to interpret due to their comparatively small numbers. For this reason, the classes were condensed into 4 categories (Negative, Somewhat Negative, Somewhat Positive, and Positive), which allowed for more accurate models.
Models were evaluated using PostResample and Confusion Matrices. With the Confusion Matrix, sensitivity and specificity were very important to determining how good the model was at predicting each class (again, this further exemplified that models were better at predicting the Negative and Positive values, while not being good at detecting Somewhat Negative or Somewhat Positive values).
Ultimately, C5.0 was chosen for each dataset, using the condensed dependent variable consisting of 4 classes. C5.0 was chosen due to its high confidence levels while still being able to run very quickly. Random Forest did perform slightly higher accuracy-wise, but took much longer to run, rendering it very inefficient. The other model-types, SVM and KKNN could not compare when it came to accuracy.



