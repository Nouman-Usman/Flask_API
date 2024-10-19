import pandas as pd
import random

# Define lawyer specialties
specialties = ["Tax", "Corporate", "Criminal", "Family", "Civil", "Immigration", "Intellectual Property", "Real Estate", "Bankruptcy", "Employment"]

# Dummy locations
locations = ["Lahore", "Karachi", "Islamabad", "Peshawar", "Quetta"]

# Lists of first and last names for generating dummy names
first_names = ["Ahmed", "Sara", "Ali", "Fatima", "Usman", "Ayesha", "Zain", "Hina", "Hassan", "Maria", "Bilal", "Anam", "Kamran", "Nida", "Imran"]
last_names = ["Khan", "Siddiqui", "Malik", "Butt", "Chaudhry", "Sheikh", "Raza", "Javed", "Nawaz", "Iqbal", "Hussain", "Farooq", "Mir", "Shah", "Qureshi"]

# Generate random ratings
def generate_rating():
    return round(random.uniform(1.0, 5.0), 1)

# Generate random experience (in years)
def generate_experience():
    return random.randint(1, 40)

# Generate random lawyer names by combining first and last names
def generate_lawyer_name():
    first_name = random.choice(first_names)
    last_name = random.choice(last_names)
    return f"{first_name} {last_name}"

# Generate a dummy dataset of lawyers
def generate_lawyer_dataset(num_lawyers=100):
    data = {
        "Lawyer Name": [],
        "Type (Specialty)": [],
        "Experience (Years)": [],
        "Ratings": [],
        "Location": []
    }
    
    for i in range(num_lawyers):
        name = generate_lawyer_name()
        specialty = random.choice(specialties)
        experience = generate_experience()
        rating = generate_rating()
        location = random.choice(locations)
        
        data["Lawyer Name"].append(name)
        data["Type (Specialty)"].append(specialty)
        data["Experience (Years)"].append(experience)
        data["Ratings"].append(rating)
        data["Location"].append(location)
    
    return pd.DataFrame(data)

# Generate the dataset
lawyer_dataset = generate_lawyer_dataset(500)

# Print the dataset
print(lawyer_dataset)

# Save the dataset to a CSV file (optional)
lawyer_dataset.to_csv("dummy_lawyer_dataset.csv", index=False)
