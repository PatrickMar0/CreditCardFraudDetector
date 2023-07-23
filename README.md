Credit Card Project

<ul>
Data Description (Kaggle Link: https://www.kaggle.com/datasets/whenamancodes/fraud-detection?resource=download)
<ul>
<li>Data needs to be downloaded since it is too big for Github</li>
<li>Our data, titled "Fraud Detection" by user Aman Chauhan, is a 250,000+ sample CSV file of credit card transactions. Each sample consists of 31 different values, the time the transaction occurred from the start of data collection, 28 hidden values (for cardholder confidentiality), the amount transferred, and if the card was fraudulent (1) or legitimate (0).</li>
<li>As with most fraud datasets, our data is heavily skewed toward non-fraudulent data (Only 0.172% is fraudulent). This makes the traditional accuracy measure the less-than-optimal choice, so we explored other measures like precision and recall.</li>
<li>Since the time variable is based on when the transaction occurred from the start of collection, it has no real relevance to the legitimacy of a transaction and can thus be cut from the training and testing of ML algorithms. All other variables are used, and since all samples are complete (all columns filled) all samples were used.</li>
</ul>
<br>

Required Libraries
<ul>
<li>Numpy</li>
<li>os</li>
<li>TensorFlow (for Keras)</li>
<li>sklearn</li>
<li>matplotlib</li>
</ul>
<br>

Code Files
<ul>
<li>Main.py: Loads the data from the Kaggle file, cutting out the titles and time column, and creates a menu to choose which to run <strong>(Run this file)</strong></li>
<li>NN.py: Applies a neural network to the Kaggle data, displaying precision and recall on test samples</li>
<li>DT.py: Applies a decision tree to the Kaggle data, displaying accuracy and precision on test samples</li>
</ul>
</ul>
