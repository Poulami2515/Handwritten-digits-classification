# Handwritten-digits-classification

A feed forward linear architecture of neural network (logistic regression model) is trained to identify handwritten digits from the MNIST dataset. The dataset consists of 28px by 28px grayscale images of handwritten digits (0 to 9) and labels for each image indicating which digit it represents.

It's quite challenging to improve the accuracy of a logistic regression model beyond approximately 87%, since the model assumes a linear relationship between pixel intensities and image labels. The best accuracy obtained using logistic regression model is 85.4%.

To improve accuracy, a random forest model has been trained. It has shown an accuracy of 96.75%.


# DEPLOYING THE MODEL

The random forest model is deployed using Streamlit. If streamlit is not pre installed, use the following command :
```bash
pip install streamlit
```
If it is pre-installed, use the following command :
```bash
pip install --upgrade streamlit
```
Assuming all other dependencies are pre-installed, download the trained models, i.e., mnist_model and random_forest_model in the same directory as app.py.
Open terminal in the same directory and run the following command :
```bash
streamlit run app.py
```












![Screenshot from 2024-08-06 14-07-20](https://github.com/user-attachments/assets/dbe5dbb5-c34f-4faa-b1a9-2424039f2705)



![Screenshot from 2024-08-06 14-08-12](https://github.com/user-attachments/assets/1bec7fe5-8f83-441a-9176-191be4277359)


































