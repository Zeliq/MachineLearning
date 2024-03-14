# import matplotlib.pyplot as plt 
# from sklearn.linear_model import LinearRegression 
# import numpy as np 
# #Given data 
# years = np.array([2013,2014,2015,2016,2017]).reshape(-1,1) 
# advertising = np.array([100000,120000,140000,160000,220000]) 
# sales = np.array([100000,120000,140000,160000,220000]) 
# #Creating and fitting the model 
# model = LinearRegression() 
# model.fit(years,sales) 
# #Predicting sale for the next year 
# next_year = np.array([[2018]]) 
# predicted_sales = model.predict(next_year) 
# print("Predicted sales for 2018 based on advertising expenditure: ", predicted_sales[0]) 
# #Plotting the linear regression line and data points 
# plt.figure(figsize=(8,6)) 
# plt.scatter(years,sales,color="blue",label = "Actual data") 
# plt.plot(years,model.predict(years),color = "red", label = "Linear regression") 
# plt.scatter(next_year,predicted_sales,color="green",label = "Predicted Point") 

# plt.title("Linear Regression: Advertising vs. Sales") 
# plt.xlabel("Year") 
# plt.ylabel("Sales") 
# plt.legend() 
# plt.grid(True) 
# plt.show()

# import numpy as np 
# import matplotlib.pyplot as plt 
# from sklearn.decomposition import PCA 
# # Load the image 
# img = plt.imread(r".\Bridal.png") 
# # Convert the image into a two-dimensional array 
# img_2d = img.reshape(img.shape[0], img.shape[1] * img.shape[2]) 
# # Apply PCA to the two-dimensional array to compress the image 
# pca = PCA(n_components=200) 
# img_pca = pca.fit_transform(img_2d) 
# # Reconstruct the compressed image 
# img_reconstructed = pca.inverse_transform(img_pca) 
# # Display the reconstructed image 
# plt.imshow(img_reconstructed.reshape(img.shape)) 
# plt.title('Reconstructed Image') 
# plt.savefig('new.jpg') 
# plt.show()


# import numpy as np 
# from scipy.linalg import svd 
# # Example ratings data (replace this with your actual data) 
# ratings_data = np.array([ 
# [5, 4, 0, 2, 1], 
# [3, 2, 0, 4, 5], 
# [0, 0, 5, 4, 3], 
# [1, 2, 3, 0, 0] 
# ]) 
# # Perform Singular Value Decomposition (SVD) 
# U, sigma, VT = svd(ratings_data) 
# # Extract the most important factors 
# num_factors_to_keep = 2  # Choose the number of factors to keep 
# important_factors = VT[:num_factors_to_keep] 
# # Print the important factors 
# print("Important factors affecting ratings:") 
# for i, factor in enumerate(important_factors): 
#     print(f"Factor {i+1}: {factor}") 
# # Calculate the reconstructed ratings matrix using the important factors 
# reconstructed_ratings=np.dot(np.dot(U[:,:num_factors_to_keep], np.diag(sigma[:num_factors_to_keep])), VT[:num_factors_to_keep]) 
# # Print the reconstructed ratings matrix 
# print("\nReconstructed ratings matrix:") 
# print(reconstructed_ratings) 


def calculate_probability(material_balance): 
    # Probability calculation based on material balance 
    total_value = sum(material_balance.values()) 
    probability = material_balance['white'] / total_value if total_value != 0 else 0.5 
    return probability 
 

# Material values of chess pieces 
piece_values = {'pawn': 1, 'knight': 3, 'bishop': 3, 'rook': 5, 'queen': 9} 

# Example material balance 
# You need to input the actual material balance from your chess position 
material_balance = {'white': 39, 'black': 42} 

# Calculate probability 
probability = calculate_probability(material_balance) 
print("Probability of white winning:", probability) 
 
