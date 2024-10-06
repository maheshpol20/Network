import sys
import time
import pulp
import pandas as pd
import math
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QProgressBar, QTextEdit
from PyQt5.QtCore import Qt, pyqtSignal, QRunnable, QThreadPool, QObject
from PyQt5.QtGui import QIcon
from pulp import COIN_CMD  # Import the COIN_CMD solver
import sys
import os

if hasattr(sys, '_MEIPASS'):
    base_path = sys._MEIPASS
else:
    base_path = os.path.abspath(".")

# Use base_path to construct paths to your resources
resource_path = os.path.join(base_path, 'your_resource_file')

# Rest of your code


class WorkerSignals(QObject):
    progress = pyqtSignal(int)
    result = pyqtSignal(pd.DataFrame)
    error = pyqtSignal(str)
    message = pyqtSignal(str)

class AllocationWorker(QRunnable):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.signals = WorkerSignals()

    def run(self):
        try:
            # Read the Excel file
            stocks_df = pd.read_excel(self.file_path, sheet_name='Stocks')
            demand_df = pd.read_excel(self.file_path, sheet_name='Demand')
            locations_df = pd.read_excel(self.file_path, sheet_name='Warehouse')

            # Add a dummy plant in locations_df
            dummy_location = pd.DataFrame([{'Branch': 'United Kingdom', 'Plant': 'UK01', 'Longitude': 0, 'Latitude': 0}])
            locations_df = pd.concat([dummy_location, locations_df], ignore_index=True)

            # Get unique SKUs
            unique_skus = demand_df['SKU'].unique()

            # Initialize a list to store results for each SKU
            results = []

            # Initialize a list to store the distance matrix for reference
            distance_matrix_reference = []

            def process_sku(sku):
                # Filter demand and stocks DataFrames for the current SKU
                sku_demand_df = demand_df[demand_df['SKU'] == sku]
                sku_stocks_df = stocks_df[stocks_df['SKU'] == sku]

                # Add stock at dummy plant in sku_stocks_df
                dummy_stock = pd.DataFrame([{'Plant': 'UK01', 'Storage Location': '2000', 'Stocks': 1000000000000000, 'SKU': sku}])
                sku_stocks_df = pd.concat([dummy_stock, sku_stocks_df], ignore_index=True)

                # Create a key by concatenating Plant and Storage Location
                sku_stocks_df['Key'] = sku_stocks_df['Plant'] + '_' + sku_stocks_df['Storage Location'].astype(str)
                sku_demand_df['Key'] = sku_demand_df['Plant'] + '_' + sku_demand_df['Storage Location'].astype(str)

                # Populate the Stocks dictionary, ensuring no NaN values
                Stocks = {row['Key']: row['Stocks'] for _, row in sku_stocks_df.iterrows() if pd.notna(row['Key']) and pd.notna(row['Stocks'])}

                # Populate the Demand dictionary, ensuring no NaN values
                Demand = {row['Key']: row['Demand'] for _, row in sku_demand_df.iterrows() if pd.notna(row['Key']) and pd.notna(row['Demand'])}

                # Populate the Locations dictionary, ensuring no NaN values
                Locations = {}
                for _, row in locations_df.iterrows():
                    plant = row['Plant']
                    longitude = row['Longitude']
                    latitude = row['Latitude']
                    if pd.notna(plant) and pd.notna(longitude) and pd.notna(latitude):
                        Locations[plant] = (longitude, latitude)

                # Calculate the distance matrix using the Haversine formula
                Distance_Matrix = {
                    (warehouse, plant): haversine(Locations[warehouse.split('_')[0]][1], Locations[warehouse.split('_')[0]][0], Locations[plant.split('_')[0]][1], Locations[plant.split('_')[0]][0])
                    for warehouse in Stocks
                    for plant in Demand
                }

                # Collect the distance matrix for reference
                for (warehouse, plant), distance in Distance_Matrix.items():
                    from_plant = warehouse.split('_')[0]
                    to_plant = plant.split('_')[0]
                    from_branch = locations_df[locations_df['Plant'] == from_plant]['Branch'].values[0]
                    to_branch = locations_df[locations_df['Plant'] == to_plant]['Branch'].values[0]
                    distance_matrix_reference.append({
                        'From Plant': from_plant,
                        'From Branch': from_branch,
                        'To Plant': to_plant,
                        'To Branch': to_branch,
                        'Distance': distance
                    })

                # Create the LP problem
                prob = pulp.LpProblem("Minimize_Transportation_Distance", pulp.LpMinimize)

                # Create decision variables for the amount to transport from each warehouse to each plant
                routes = pulp.LpVariable.dicts("Route", Distance_Matrix, lowBound=0, cat='Continuous')

                # Objective function: Minimize the total transportation distance
                prob += pulp.lpSum([Distance_Matrix[w_p] * routes[w_p] for w_p in Distance_Matrix]), "Total_Transportation_Distance"

                # Constraints: The total quantity shipped from each warehouse cannot exceed the stock
                for warehouse in Stocks:
                    prob += pulp.lpSum([routes[(warehouse, plant)] for plant in Demand]) <= Stocks[warehouse], f"Stock_{warehouse}"

                # Constraints: The total quantity received by each plant must equal the demand
                for plant in Demand:
                    prob += pulp.lpSum([routes[(warehouse, plant)] for warehouse in Stocks]) == Demand[plant], f"Demand_{plant}"

                # Constraint: The sum of the quantity received by all plants must not exceed the sum of the stock available at all warehouses
                prob += pulp.lpSum([routes[w_p] for w_p in Distance_Matrix]) <= sum(Stocks.values()), "Total_Stock"

                # Solve the problem using the COIN_CMD solver
                cbc_path = os.path.join(sys._MEIPASS, 'cbc.exe') if hasattr(sys, '_MEIPASS') else 'cbc'
                prob.solve(COIN_CMD(path=cbc_path))

                # Check if the solver found an optimal solution
                if prob.status != pulp.LpStatusOptimal:
                    raise ValueError("Solver did not find an optimal solution")

                # Collect the results for the current SKU
                for w_p in Distance_Matrix:
                    allocation = routes[w_p].varValue
                    if allocation is not None and allocation > 0:  # Only include rows where Quantity > 0
                        from_plant, from_storage = w_p[0].split('_')
                        to_plant, to_storage = w_p[1].split('_')
                        from_row = sku_stocks_df[sku_stocks_df['Key'] == w_p[0]].iloc[0]
                        to_row = sku_demand_df[sku_demand_df['Key'] == w_p[1]].iloc[0]
                        decision = "IBT"
                        if from_plant != to_plant:
                            if from_storage == "2000":
                                decision = "IBT"                
                            else:
                                decision = "ICT and IBT"
                        else:
                            decision = "ICT"

                        results.append({
                            'SKU': sku,
                            'From Region': from_row['Region'],
                            'From Plant': from_plant,
                            'From Storage Location': from_storage,
                            'From Branch': locations_df[locations_df['Plant'] == from_plant]['Branch'].values[0],
                            'To Region': to_row['Region'],
                            'To Plant': to_plant,
                            'To Storage Location': to_storage,
                            'To Branch': locations_df[locations_df['Plant'] == to_plant]['Branch'].values[0],
                            'Quantity': allocation,
                            'Distance': Distance_Matrix[w_p],
                            'Decision': decision
                        })

                        if from_plant != 'UK01':
                            from_branch = locations_df[locations_df['Plant'] == from_plant]['Branch'].values[0]
                            to_branch = locations_df[locations_df['Plant'] == to_plant]['Branch'].values[0]
                            demand = Demand[w_p[1]]
                            self.signals.message.emit(f"{sku}:- {allocation}/{demand} Allocated From: {from_branch}_{from_storage}, To: {to_branch}_{to_storage}")

            for i, sku in enumerate(unique_skus):
                process_sku(sku)
                self.signals.progress.emit(int((i + 1) / len(unique_skus) * 100))

            # Convert the results to a DataFrame
            results_df = pd.DataFrame(results)

            # Filter out rows where 'From Plant' is 'UK01'
            results_df = results_df[results_df['From Plant'] != 'UK01']

            # Convert the distance matrix reference to a DataFrame and drop duplicates
            distance_matrix_df = pd.DataFrame(distance_matrix_reference).drop_duplicates()

            # Remove rows where 'From Plant' or 'To Plant' is 'UK01'
            distance_matrix_df = distance_matrix_df[(distance_matrix_df['From Plant'] != 'UK01') & (distance_matrix_df['To Plant'] != 'UK01')]

            # Save the demand, stocks, final solution, and distance matrix to an Excel file
            with pd.ExcelWriter('Result.xlsx') as writer:
                # Save the demand and stocks DataFrames
                demand_df.to_excel(writer, sheet_name='Demand', index=False)
                stocks_df.to_excel(writer, sheet_name='Stocks', index=False)
                
                # Save the final solution to a DataFrame
                results_df.to_excel(writer, sheet_name='Solution', index=False)
                
                # Save the distance matrix reference to a DataFrame
                distance_matrix_df.to_excel(writer, sheet_name='Distance Matrix', index=False)

            self.signals.result.emit(results_df)
        except Exception as e:
            self.signals.error.emit(str(e))

class AllocationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stocks Allocation Tool ⚙️")
        self.setWindowIcon(QIcon('logo.png'))  # Ensure 'logo.png' is in your project directory
        self.setGeometry(100, 100, 725, 662)  # Increased by 25%
        
        self.Input_file_path = ""
        
        self.title_label = QLabel('<span style="font-size: 32px; font-weight: bold; color: #2E8B57;">Stocks Allocation Tool</span>', self)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.about_button = QPushButton("📝About", self)
        self.about_button.setStyleSheet("font-size: 20px; background-color: #2E8B57; color: white;")  # Increased by 25%
        self.about_button.setFixedWidth(90)  # Increased by 25%
        self.about_button.clicked.connect(self.show_about_message)
        
        self.browse_button = QPushButton("📂Browse Input File", self)
        self.browse_button.setStyleSheet("font-size: 20px; background-color: #2E8B57; color: white;")  # Increased by 25%
        self.browse_button.setFixedWidth(195)  # Increased by 25%
        self.browse_button.clicked.connect(self.browse_Input_file)
        
        self.Allocation_button = QPushButton("🧠Calculate ", self)
        self.Allocation_button.setStyleSheet("font-size: 20px; background-color: #004282; color: white;")  # Increased by 25%
        self.Allocation_button.setFixedWidth(175)  # Increased by 25%
        self.Allocation_button.clicked.connect(self.calculate_Allocation)
        
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setStyleSheet("QProgressBar {border: 2px solid #004282; border-radius: 5px; text-align: center; height: 37px;} QProgressBar::chunk {background-color: #2E8B57; width: 25px;}")  # Increased by 25%
        self.progress_bar.setValue(0)
        
        self.message_box = QTextEdit(self)
        self.message_box.setReadOnly(True)
        self.message_box.setStyleSheet("font-size: 17px; color: green; background-color: #E0FFE0; padding: 6px;")  # Increased by 25%
        self.message_box.setFixedHeight(350)  # Adjust the height as needed
        
        self.crompton_label = QLabel("Optimizing Resource for Brighter Tomorrow 🚀", self)
        self.crompton_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.crompton_label.setStyleSheet("font-size: 17px; color: #004282;font-weight: bold; font-style: italic;")  # Adjust the style as needed

        self.comment_label = QLabel("Developed by Crompton SCM Team", self)
        self.comment_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.comment_label.setStyleSheet("font-size: 17px; color: grey; ")  # Increased by 25%
        
        layout = QVBoxLayout()
        layout.addWidget(self.title_label)
        layout.addSpacing(25)  # Increased by 25%
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.about_button)
        button_layout.addWidget(self.browse_button)
        button_layout.addWidget(self.Allocation_button)
        layout.addLayout(button_layout)
        
        layout.addSpacing(25)  # Increased by 25%
        layout.addWidget(self.progress_bar)
        layout.addSpacing(25)  # Increased by 25%
        layout.addWidget(self.message_box)
        layout.addStretch()
        layout.addWidget(self.crompton_label)
        layout.addWidget(self.comment_label)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        self.show_about_message()
        
    def show_about_message(self):
        self.message_box.setHtml("Do's & Don't:-<br>"
                                "1. Maintain Longitude and Latitude in Warehouse sheet for all warehouses.<br>"
                                "2. Enter regions correctly.<br>"
                                "3. Blank cells not allowed in input excel, especially Demand & Stock columns.<br>"
                                "4. Do not change any table headers in Input File.<br>"
                                "<br>"
                                "Please contact <b>Mahesh Pol(8149598176)</b> for any errors/bugs/logic modifications")
        
    def browse_Input_file(self):
        file_dialog = QFileDialog()
        self.Input_file_path, _ = file_dialog.getOpenFileName(self, "Select Input File", "", "Excel Files (*.xlsx)")
        if self.Input_file_path:
            self.message_box.setText(f"Selected file: {self.Input_file_path}")
    
    def calculate_Allocation(self):
        if not self.Input_file_path:
            self.message_box.setText("Please select a Input file first.")
            return
        
        self.start_time = time.time()  # Start the timer when the button is clicked
        
        self.worker = AllocationWorker(self.Input_file_path)
        self.worker.signals.progress.connect(self.update_progress)
        self.worker.signals.result.connect(self.save_results)
        self.worker.signals.error.connect(self.show_error)
        self.worker.signals.message.connect(self.update_message)
        QThreadPool.globalInstance().start(self.worker)
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    def update_message(self, message):
        self.message_box.append(message)
        self.message_box.verticalScrollBar().setValue(self.message_box.verticalScrollBar().maximum())
    
    def save_results(self, df_new):
        try:
            end_time = time.time()
            elapsed_time = end_time - self.start_time
            self.message_box.append(f"<b>Allocation completed in {elapsed_time:.2f} seconds. Results saved to 'Result.xlsx'</b>")
            self.update_progress(100)  # Ensure progress bar reaches 100%
        
        except Exception as e:
            self.message_box.setText(f"An error occurred while saving results: {str(e)}")
    
    def show_error(self, error_message):
        self.message_box.setText(f"An error occurred: {error_message}")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Radius of the Earth in kilometers

    # Convert latitudes and longitudes from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Differences in coordinates
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance in kilometers
    distance = R * c

    return distance

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AllocationApp()
    window.show()
    sys.exit(app.exec_())