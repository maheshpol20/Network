#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
import time
from collections import defaultdict
import copy


# In[2]:


initial_time = time.time()


# In[3]:


# Specify the filename of your Excel file
filename = 'Data Allocation 1.xlsx'  # Replace with your actual filename
rounds = 10


# In[4]:


# Constants
stages = 4
minimum_truck_load = 15


# In[5]:


# Network
# SKU	Depot	Parent Code	Active	Alternate Plant
start_time = time.time()

sheet_name = "Network"
network = pd.read_excel(filename, sheet_name = sheet_name)
network_df = pd.DataFrame(network)
network_df.columns = network_df.columns.str.strip()

convert_dict = {
    'SKU': str,
    'Depot': str,
    'Parent Code': str,
    'Active': 'int64',
    'Alternate Plant': str
}

network_df = network_df.astype(convert_dict)
network_df = network_df.drop_duplicates()
network_df.fillna(0, inplace = True)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.3f} seconds")


# In[6]:


# Demand
start_time = time.time()

sheet_name = "Demand"
demand = pd.read_excel(filename, sheet_name = sheet_name)
demand_df = pd.DataFrame(demand)
demand_df.columns = demand_df.columns.str.strip()

# print(demand_df.dtypes)

convert_dict = {
    'SKU': str,
    'Depot': str,
    'Safety Stock': 'int64',
    'M1': 'int64',
    'M2': 'int64',
}

demand_df = demand_df.astype(convert_dict)

demand_df.fillna(0, inplace = True)

# print(demand_df.dtypes)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.3f} seconds")


# In[7]:


# StockSITSTO
start_time = time.time()

sheet_name = "StockSITSTO"
sitsto = pd.read_excel(filename, sheet_name = sheet_name)
sitsto_df = pd.DataFrame(sitsto)
sitsto_df.columns = sitsto_df.columns.str.strip()

# print(type(sitsto_df['STO']))

convert_dict = {
    'SKU': str,
    'Depot': str,
    'Stock+SIT': 'int64',
    'STO': 'int64',
}

sitsto_df = sitsto_df.astype(convert_dict)

# print(type(sitsto_df['STO']))
sitsto_df.fillna(0, inplace = True)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.3f} seconds")


# In[8]:


# SKU Data
start_time = time.time()

sheet_name = "SKU Data"
sku_wt = pd.read_excel(filename, sheet_name = sheet_name)
sku_df = pd.DataFrame(sku_wt)
sku_df.columns = sku_df.columns.str.strip()

# print(sku_df.dtypes)

convert_dict = {
    'SKU': str,
    'MOQ': 'int64',
    'Pack Size': 'float64',
    'Gross Weight (kg)': 'float64',
}

sku_df = sku_df.astype(convert_dict)

sku_df.fillna('N/A', inplace = True)

# print(sku_df.dtypes)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.3f} seconds")


# In[9]:


# PSO
start_time = time.time()

sheet_name = "PSO"
pso = pd.read_excel(filename, sheet_name = sheet_name)
pso_df = pd.DataFrame(pso)
pso_df.columns = pso_df.columns.str.strip()

convert_dict = {
    'SKU': str,
    'Depot': str,
    'PSO': 'int64',
}

pso_df = pso_df.astype(convert_dict)

pso_df.fillna(0, inplace = True)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.3f} seconds")


# In[10]:


# Depot Name
# Depot	Depot Name	Hub Code	Hub Name	Zone


start_time = time.time()

sheet_name = "Depot Data"
depot_name = pd.read_excel(filename, sheet_name = sheet_name)
depot_df = pd.DataFrame(depot_name)
depot_df.columns = depot_df.columns.str.strip()

convert_dict = {
    'Depot': str,
    'Depot Name': str,
    'Zone': str,
}

depot_df = depot_df.astype(convert_dict)

depot_df.fillna('#N/A', inplace = True)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.3f} seconds")


# In[11]:


# Plant Name
# Plant Code	Plant Location Name

start_time = time.time()

sheet_name = "Plant Data"
plant_name = pd.read_excel(filename, sheet_name = sheet_name)
plant_name_df = pd.DataFrame(plant_name)
plant_name_df.columns = plant_name_df.columns.str.strip()

convert_dict = {
    'Plant Code': str,
    'Plant Location Name': str,
}

plant_name_df = plant_name_df.astype(convert_dict)

plant_name_df.fillna('#N/A', inplace = True)


end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.3f} seconds")


# In[12]:


# Remarks
# Start Weight (tonne)	End Weight (tonne)	Remark

start_time = time.time()

sheet_name = "Remarks"
remarks_data = pd.read_excel(filename, sheet_name = sheet_name)
remarks_data_df = pd.DataFrame(remarks_data)
remarks_data_df.columns = remarks_data_df.columns.str.strip()
remarks_data_df.fillna('#N/A', inplace = True)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.3f} seconds")


# In[13]:


# Pallet MOQ
start_time = time.time()

sheet_name = "Pallet MOQ"
pallet = pd.read_excel(filename, sheet_name = sheet_name)
pallet_df = pd.DataFrame(pallet)
pallet_df.columns = pallet_df.columns.str.strip()

convert_dict = {
    'SKU': str,
    'Depot': str,
    'Pallet MOQ': 'int64',
}

pallet_df = pallet_df.astype(convert_dict)

pallet_df.fillna(0, inplace = True)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.3f} seconds")


# In[14]:


# Utility function to create dictionary

def multi_dict(K, type):
    if K == 1:
        return defaultdict(type)
    else:
        return defaultdict(lambda: multi_dict(K-1, type))


# In[15]:


# SKU vs Plant Dataframe
start_time = time.time()

# sku_plant_df = network_df.copy(deep = True)
network_df.drop(network_df[network_df['Active'] == 0].index, inplace=True)

sku_plant_df = network_df[['SKU', 'Parent Code']].copy(deep = True)

sku_plant_df = sku_plant_df.drop_duplicates()

sku_plant_df['Depot'] = sku_plant_df['Parent Code']
sku_plant_df['Parent Code'] = sku_plant_df['Parent Code'].astype(str)
sku_plant_df['Parent Code'] = sku_plant_df['Parent Code'] + 'P'

sku_alternate_plant_df = network_df[['SKU', 'Alternate Plant']].copy(deep = True)

sku_alternate_plant_df = sku_alternate_plant_df.drop_duplicates()

sku_alternate_plant_df['Depot'] = sku_alternate_plant_df['Alternate Plant']
sku_alternate_plant_df['Alternate Plant'] = sku_alternate_plant_df['Alternate Plant'].astype(str)
sku_alternate_plant_df['Alternate Plant'] = sku_alternate_plant_df['Alternate Plant'] + 'P'
sku_alternate_plant_df['Parent Code'] = sku_alternate_plant_df['Alternate Plant']
sku_alternate_plant_df.drop(columns=['Alternate Plant'], inplace=True)

final_network_df = pd.concat([network_df, sku_plant_df, sku_alternate_plant_df], ignore_index=True)
final_network_df = final_network_df.loc[(final_network_df['Parent Code'] != '0P') & (final_network_df['Parent Code'] != '0')]
network_df = final_network_df.copy(deep = True)
network_df = network_df.drop_duplicates()

# print(len(network_df))
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.3f} seconds")


# In[16]:


# Dictionary for Parent Plants

parent_dict = {}
alternate_parent_dict = {}
for index, row in network_df.iterrows():
    depot = row['Depot']
    sku = row['SKU'] 
    parent = row['Parent Code']
    alternate_parent = row['Alternate Plant']
    parent_dict[(sku,depot)] = parent
    alternate_parent_dict[(sku, depot, parent)] = alternate_parent 


# In[17]:


# Dictionary for SKU Weight
# SKU	MOQ	Pack Size	Gross Weight (kg)	Net Weight (kg)	Category

start_time = time.time()

sku_wt_dict = {}
moq_dict = {}
category_dict = {}
desc_dict = {}
pack_dict = {}
for index, row in sku_df.iterrows():
    try:
        sku = row['SKU']
        try:
            wt = max(row['Gross Weight (kg)'], 0)
        except:
            wt = 1
        try:
            moq = max(row['MOQ'], 1)
        except:
            moq = 1

        try:
            category = row['Category']
        except:
            category = "C"

        try:
            pack = max(row['Pack Size'], 0)
        except:
            pack = 1

        try:
            desc = row['SKU Desc']
        except:    
            desc = "N/A"

        pack_dict[sku] = pack
        if pack == 0:
            pack_dict[sku] = 1
            pack = 1
        sku_wt_dict[sku] = max(wt, 0)
        if sku_wt_dict[sku] == 0:
            sku_wt_dict[sku] = pack
        moq_dict[sku] = max(moq,1)
        category_dict[sku] = category
        desc_dict[sku] = desc 
    except Exception as e:
        sku_wt_dict[sku] = pack      
        moq_dict[sku] = 1
        category_dict[sku] = "C"
        desc_dict[sku] = "N/A"
        print("Error:", e, " in sku_wt at SKU:", sku)

sku_df_list = sku_df['SKU'].tolist()

sku_list = network_df['SKU']
sku_list = sku_list.drop_duplicates()

for sku in sku_list:
    if sku not in sku_df_list:
        sku_wt_dict[sku] = 1
        moq_dict[sku] = 1
        category_dict[sku] = "C"
        desc_dict[sku] = "N/A"
        pack_dict[sku] = 1

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.3f} seconds")


# In[18]:


# Dictionary for Total Demand
# SKU	Depot	M1	M2	M3


start_time = time.time()

demand_dict = multi_dict(3, float)
demand_map = multi_dict(2, float)

for index, row in demand_df.iterrows():
    try:
        sku = row['SKU']
        depot = row['Depot']
        try:
            demand_m1 = max(row['Safety Stock'], 0)
        except:
            demand_m1 = 0

        try:
            demand_m2 = max(row['M1'], 0)
        except:
            demand_m2 = 0

        try:
            demand_m3 = max(row['M2'], 0)
        except:
            demand_m3 = 0

        demand_dict[1][sku][depot] = max(demand_m1,0)
        demand_dict[2][sku][depot] = max(demand_m2,0) 
        demand_dict[3][sku][depot] = max(demand_m3,0) 

        demand_map[1][(sku, depot)] = max(demand_m1, 0)
        demand_map[2][(sku, depot)] = max(demand_m2, 0)
        demand_map[3][(sku, depot)] = max(demand_m3, 0)

    except Exception as e:  
        print("Error:", e, " in Demand at SKU:", sku)
# count = 0
for index, row in network_df.iterrows():
    sku = row['SKU']
    depot = row['Depot']
    if (sku, depot) not in demand_map[1].keys():
        demand_map[1][(sku, depot)] = 0
        demand_dict[1][sku][depot] = 0
    if (sku, depot) not in demand_map[2].keys():
        demand_map[2][(sku, depot)] = 0
        demand_dict[2][sku][depot] = 0
    if (sku, depot) not in demand_map[3].keys():
        demand_map[3][(sku, depot)] = 0
        demand_dict[3][sku][depot] = 0

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.3f} seconds")


# In[19]:


# Dictionary for Stock SIT STO
# Key	SKU	Depot	Stock+SIT	STO

start_time = time.time()

sit_dict = multi_dict(2, float)
sto_dict = multi_dict(2, float)
sit_map = {}
sto_map = {}
soh_map = {}
for index, row in sitsto_df.iterrows():
    try:
        sku = row['SKU']
        depot = row['Depot']

        try:
            sit = max(row['Stock+SIT'], 0)
        except:
            sit = 0

        try:
            sto = max(row['STO'], 0)
        except:
            sto = 0

        sit_dict[sku][depot] = max(sit,0)
        sto_dict[sku][depot] = max(sto,0) 

        sit_map[(sku, depot)] = max(sit,0)
        sto_map[(sku, depot)] = max(sto, 0)
        soh_map[(sku, depot)] = max(sit+sto, 0)
    except Exception as e:
        sit_dict[sku][depot] = 0        
        sto_dict[sku][depot] = 0
        print("Error:", e, " in Stock SIT and STO at SKU:", sku)

for index, row in network_df.iterrows():
    sku = row['SKU']
    depot = row['Depot']
    if (sku, depot) not in sit_map.keys():
        sit_map[(sku, depot)] = 0
        sit_dict[sku][depot] = 0
    if (sku, depot) not in sto_map.keys():
        sto_map[(sku, depot)] = 0
        sto_dict[sku][depot] = 0
    if (sku, depot) not in soh_map.keys():
        soh_map[(sku, depot)] = 0
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.3f} seconds")


# In[20]:


# # Dictionary for PSO
# Key	SKU	Depot	PSO

start_time = time.time()

for index, row in pso_df.iterrows():
    try:
        sku = row['SKU']
        depot = row['Depot']
        try:
            pso = max(row['PSO'], 0)
        except:
            pso = 0

        demand_dict[0][sku][depot] = max(pso, 0) 
        demand_map[0][(sku, depot)] = max(pso, 0)      
    except Exception as e:
        demand_dict[0][sku][depot] = 0        
        print("Error:", e, " in pso at SKU:", sku)

for index, row in network_df.iterrows():
    sku = row['SKU']
    depot = row['Depot']
    if (sku, depot) not in demand_map[0].keys():
        # print(sku, depot)
        demand_map[0][(sku, depot)] = 0
        demand_dict[0][sku][depot] = 0

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.3f} seconds")


# In[21]:


# Dictionary for Depot Data
# Depot	Depot Name	Hub Code	Hub Name	Zone

start_time = time.time()

depot_dict = multi_dict(2, str)
depot_dict_map = multi_dict(2, str)
for index, row in depot_df.iterrows():
    try:
        depot_code = row['Depot']

        try:
            name = row['Depot Name']
        except:
            name = "N/A"

        try:
            zone = row['Zone']
        except:
            zone = "N/A"

        try:
            hub_code = row['Hub Code']
        except:
            hub_code = "N/A"

        try:
            hub_name = row['Hub Name']
        except:
            hub_name = "N/A"

        depot_dict[depot_code]['name'] = name
        depot_dict[depot_code]['zone'] = zone
        depot_dict[depot_code]['hub code'] = hub_code
        depot_dict[depot_code]['hub name'] = hub_name

        depot_dict_map['zone'][depot_code] = zone
        depot_dict_map['name'][depot_code] = name
        depot_dict_map['hub name'][depot_code] = hub_name
        depot_dict_map['hub code'][depot_code] = hub_code
    except Exception as e:
        print("Error", e)

depot_df_list = depot_df['Depot'].tolist()

depot_list = network_df['Depot']
depot_list = depot_list.drop_duplicates()

for depot_code in depot_list:
    if depot_code not in depot_df_list:
        depot_dict[depot_code]['name'] = "N/A"
        depot_dict[depot_code]['zone'] = "N/A"
        depot_dict[depot_code]['hub code'] = "N/A"
        depot_dict[depot_code]['hub name'] = "N/A"

        depot_dict_map['zone'][depot_code] = "N/A"
        depot_dict_map['name'][depot_code] = "N/A"
        depot_dict_map['hub name'][depot_code] = "N/A"
        depot_dict_map['hub code'][depot_code] = "N/A"

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.3f} seconds")


# In[22]:


# Dictionary for Plant Name

start_time = time.time()

plant_name_dict = {}
# alternate_parent_dict[(sku, depot, parent_code)]_dict = {}
for index, row in plant_name_df.iterrows():
    try:
        plant_code = row['Plant Code']
        try:
            plant_location_name = row['Plant Location Name']
        except:
            plant_location_name = "N/A"

        depot_dict[plant_code]['zone'] = "Plant"
        depot_dict_map['zone'][plant_code] = "Plant"
        plant_name_dict[plant_code] = plant_location_name
    except Exception as e:
        plant_name_dict[plant_code] = "N/A"
        print("Error:", e, " in Plant Name at Plant Code", plant_code)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.3f} seconds")


# In[23]:


# # Dictionary for Pallet MOQ
# Depot	SKU	Pallet MOQ

start_time = time.time()

pallet_dict = multi_dict(2, float)
pallet_map = {}
for index, row in pallet_df.iterrows():
    try:
        sku = row['SKU']
        depot = row['Depot']
        try:
            pallet_moq = max(row['Pallet MOQ'], 1)
        except:
            pallet_moq = 1

        pallet_dict[sku][depot] = max(pallet_moq,0)
        pallet_map[(sku, depot)] = max(pallet_moq, 0)

    except Exception as e:  
        print("Error:", e, " in Pallet at SKU:", sku)

for index, row in network_df.iterrows():
    sku = row['SKU']
    depot = row['Depot']
    if (sku, depot) not in pallet_map.keys():
        try:
            pallet_map[(sku, depot)] = moq_dict[sku]
            pallet_dict[sku][depot] = moq_dict[sku]
        except:
            pallet_map[(sku, depot)] = 1
            pallet_dict[sku][depot] = 1

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.3f} seconds")


# In[24]:


# Creating Allocation Dictionary

total_tonnage_allocation_dict = multi_dict(2, float)
total_hub_tonnage_allocation_dict = multi_dict(2, float)


# In[25]:


# Each Stage Allocation mapping

stage_allocation_map = multi_dict(4, float)


# In[26]:


# For extra allocation

dispatch_dict = multi_dict(3, list)


# In[27]:


# Allocation Dataframe

start_time = time.time()

stage_dict = {
    0: 'PSO',
    1: 'Safety Stock',
    2: 'M1',
    3: 'M2',
}

column_names = [
    'Key',
    'SKU Plant Key',
    'SKU Code',
    'SKU Desc',
    'Category',
    'Depot',
    'Depot Name',
    'Zone',
    'Parent Code',
    'Parent location Name',
    'Panipat SKUs',
    'Alternate Plant',

    'MOQ',
    'Pack Size',
    'Stock+SIT',
    'STO',

    'Total Demand (PSO)',
    'Req Qty (PSO)',
    'Allocation (PSO)',
    'Shortage Qty (PSO)',
    'Plant Leftover Stock (PSO)',

    'Cross Req Qty (PSO)',
    'Cross Allocation (PSO)',
    'Final Shortage Qty (PSO)',
    'Final Plant Leftover Stock (PSO)',

    'Total Demand (Safety Stock)',
    'Req Qty (Safety Stock)',
    'Allocation (Safety Stock)',
    'Shortage Qty (Safety Stock)',
    'Plant Leftover Stock (Safety Stock)',

    'Cross Req Qty (Safety Stock)',
    'Cross Allocation (Safety Stock)',
    'Final Shortage Qty (Safety Stock)',
    'Final Plant Leftover Stock (Safety Stock)',

    'Total Demand (M1)',
    'Req Qty (M1)',
    'Allocation (M1)',
    'Shortage Qty (M1)',
    'Plant Leftover Stock (M1)',

    'Cross Req Qty (M1)',
    'Cross Allocation (M1)',
    'Final Shortage Qty (M1)',
    'Final Plant Leftover Stock (M1)',

    'Total Demand (M2)',
    'Req Qty (M2)',
    'Allocation (M2)',
    'Shortage Qty (M2)',
    'Plant Leftover Stock (M2)',

    'Cross Req Qty (M2)',
    'Cross Allocation (M2)',
    'Final Shortage Qty (M2)',
    'Final Plant Leftover Stock (M2)',

    'Final Depot Stock',
    'Final Allocation',
    'Final Cross Allocation',
    'Total Allocation',
]
allocation_df = pd.DataFrame(columns=column_names)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.3f} seconds")


# In[28]:


# Allocations

start_time = time.time()

# Allocation
req_qty_map = multi_dict(2, float)

allocation_dict = multi_dict(4, float)
allocation_map = multi_dict(2, float)

shortage_dict = multi_dict(4, float)
shortage_map = multi_dict(2, float)    

leftover_stock_dict = multi_dict(3, float)
leftover_stock_map = multi_dict(2, float)

tonnage_allocation_dict = multi_dict(3, float)
hub_tonnage_allocation_dict = multi_dict(3, float)

tonnage_allocation_dict = multi_dict(3, float)
hub_tonnage_allocation_dict = multi_dict(3, float)

# Cross Allocation
cross_req_qty_map = multi_dict(2, float)

cross_allocation_dict = multi_dict(4, float)
cross_allocation_map = multi_dict(2, float)

cross_shortage_dict = multi_dict(4, float)
cross_shortage_map = multi_dict(2, float)

cross_leftover_stock_dict = multi_dict(3, float)
cross_leftover_stock_map = multi_dict(3, float)

tonnage_cross_allocation_dict = multi_dict(3, float)
hub_tonnage_cross_allocation_dict = multi_dict(3, float)

leftover_stock_dict[0] = copy.deepcopy(sit_dict)
leftover_stock_map[0] = copy.deepcopy(sit_map)

for stage in range(stages):
    if stage > 0:
        leftover_stock_dict[stage] = copy.deepcopy(cross_leftover_stock_dict[stage-1])
        leftover_stock_map[stage] = copy.deepcopy(cross_leftover_stock_map[stage-1])
    for round in range(1,rounds+1):

        # ------------ PSO Req Qty ----------------------------------------------------
        req_qty_dict = multi_dict(3, float)
        ceil_req_qty_dict = multi_dict(3, float)

        for index, row in network_df.iterrows():
            try:
                sku = row['SKU']
                depot = row['Depot']
                parent_code = row['Parent Code']
                if parent_code == '0':
                    continue

                try:
                    zone = depot_dict[depot]['zone']
                except:
                    zone = "#N/A"
                    depot_dict[depot]['zone'] = "#N/A"
                if zone != "Plant":
                    sit = 0
                    sto = 0
                    total_demand = 0
                    for i in range(stage+1):
                        total_demand += demand_dict[i][sku][depot]

                    try:
                        sit = sit_dict[sku][depot]
                    except Exception as e:
                        print("Error", e)
                        sit = 0
                        sit_dict[sku][depot] = 0
                        sit_map[(sku, depot)] = 0
                    try: 
                        sto = sto_dict[sku][depot]
                    except Exception as e:
                        print("Error", e)
                        sto = 0
                        sto_dict[sku][depot] = 0
                        sto_map[(sku, depot)] = 0

                    if round == 1:
                        allocation_dict[stage][sku][parent_code][depot] = 0
                    allocated = allocation_dict[stage][sku][parent_code][depot]
                    for i in range(stage):
                        allocated += allocation_dict[i][sku][parent_code][depot] + cross_allocation_dict[i][sku][parent_code][depot]

                    req_qty_dict[sku][parent_code][depot] = max(math.ceil(total_demand*round/rounds) - sit - sto - allocated, 0)
                    ceil_req_qty_dict[sku][parent_code][depot] = math.ceil(req_qty_dict[sku][parent_code][depot]/pallet_dict[sku][depot])*pallet_dict[sku][dict]

                    req_qty_map[stage][(sku, parent_code, depot)] = max(total_demand - sit - sto - allocated + allocation_dict[stage][sku][parent_code][depot], 0)
            except Exception as e:
                req_qty_dict[sku][parent_code][depot] = 0
                print("Error in calculating Req Qty:", e, " at SKU:", sku)

        # ------------- Demand Ratio (PSO) ----------------------------------------------------------------

        demand_ratio_dict = multi_dict(3, float)
        total_req_qty_dict = multi_dict(2, float)

        demand_ratio_map = {}

        for index, row in network_df.iterrows():
            try:
                sku = row['SKU']
                depot = row['Depot']
                parent_code = row['Parent Code']

                if parent_code == '0':
                    continue

                try:
                    zone = depot_dict[depot]['zone']
                except:
                    zone = "#N/A"
                    depot_dict[depot]['zone'] = "#N/A"
                if zone != "Plant":
                    total_req_qty_dict[sku][parent_code] = sum(req_qty_dict[sku][parent_code].values())
                    if total_req_qty_dict[sku][parent_code] != 0:
                        demand_ratio_dict[sku][parent_code][depot] = req_qty_dict[sku][parent_code][depot] / total_req_qty_dict[sku][parent_code]
                    else:
                        demand_ratio_dict[sku][parent_code][depot] = 0
                demand_ratio_map[(sku, parent_code, depot)] = demand_ratio_dict[sku][parent_code][depot]
            except Exception as e:
                print("Error in calculating PSO Demand Ratio", e, " at SKU:", sku)
                demand_ratio_dict[sku][parent_code][depot] = 0

        # --------------- Allocation ----------------------------------------------------   

        leftover_stock_dict_1 = copy.deepcopy(leftover_stock_dict[stage])
        leftover_stock_map_1 = copy.deepcopy(leftover_stock_map[stage])

        for index, row in network_df.iterrows():
            try:
                sku = row['SKU']
                depot = row['Depot']
                parent_code = row['Parent Code']

                if parent_code == '0':
                    continue

                try:
                    zone = depot_dict[depot]['zone']
                except:
                    zone = "#N/A"
                    depot_dict[depot]['zone'] = "#N/A"
                if zone != "Plant":            
                    allocation = min(req_qty_dict[sku][parent_code][depot], leftover_stock_dict_1[sku][parent_code]*demand_ratio_dict[sku][parent_code][depot], leftover_stock_dict[stage][sku][parent_code])
                    ceil_allocation = math.ceil(allocation/pallet_dict[sku][depot])*pallet_dict[sku][depot]        
                    allocation = math.floor(allocation/pallet_dict[sku][depot])*pallet_dict[sku][depot]

                    if ((allocation == 0 and ceil_allocation <= 2*req_qty_dict[sku][parent_code][depot]) or leftover_stock_dict_1[sku][parent_code] >= sum(ceil_req_qty_dict[sku][parent_code].values())) and ceil_allocation <= leftover_stock_dict[stage][sku][parent_code]:
                        allocation = ceil_allocation

                    if allocation != 0:
                        stage_allocation_map[stage][(depot, parent_code)][category_dict[sku]][sku] += allocation
                        dispatch_dict[(depot, parent_code)][category_dict[sku]][sku] = 1
                    try:
                        hub = depot_dict[depot]['hub code']

                        total_tonnage_allocation_dict[depot][parent_code] = total_tonnage_allocation_dict[depot][parent_code] + (allocation*sku_wt_dict[sku]/1000)
                        tonnage_allocation_dict[stage][depot][parent_code] = tonnage_allocation_dict[stage][depot][parent_code] + (allocation*sku_wt_dict[sku]/1000)

                        total_hub_tonnage_allocation_dict[hub][parent_code] = total_hub_tonnage_allocation_dict[hub][parent_code] + (allocation*sku_wt_dict[sku]/1000)
                        hub_tonnage_allocation_dict[stage][hub][parent_code] = hub_tonnage_allocation_dict[stage][hub][parent_code] + (allocation*sku_wt_dict[sku]/1000)
                    except Exception as e:
                        print("Error", e)
                        total_tonnage_allocation_dict[depot][parent_code] = allocation*sku_wt_dict[sku]/1000
                        tonnage_allocation_dict[stage][depot][parent_code] = allocation*sku_wt_dict[sku]/1000
                        total_hub_tonnage_allocation_dict[depot][hub] = (allocation*sku_wt_dict[sku]/1000)
                        hub_tonnage_allocation_dict[stage][depot][hub] = (allocation*sku_wt_dict[sku]/1000)

                    allocation_dict[stage][sku][parent_code][depot] += allocation    
                    allocation_map[stage][(sku, parent_code, depot)] = allocation_dict[stage][sku][parent_code][depot]               

                    leftover_stock_dict[stage][sku][parent_code] = leftover_stock_dict[stage][sku][parent_code] - allocation
                    leftover_stock_map[stage][(sku, parent_code)] = leftover_stock_dict[stage][sku][parent_code]

                    shortage_dict[stage][sku][parent_code][depot] = max(req_qty_dict[sku][parent_code][depot] - allocation,0)
                    shortage_map[stage][(sku, parent_code, depot)] = shortage_dict[stage][sku][parent_code][depot] 

            except Exception as e:
                print("Error in calculating Allocation (PSO)", e, " at SKU:", sku, "Depot:", depot, "Zone:", zone)
                allocation_dict[stage][sku][parent_code][depot] = 0
                allocation_map[stage][(sku, parent_code, depot)] = 0

    cross_leftover_stock_dict[stage] = copy.deepcopy(leftover_stock_dict[stage])
    cross_leftover_stock_map[stage] = copy.deepcopy(leftover_stock_map[stage])

    for round in range(1, rounds+1):

        #---------------- Cross PSO Req Qty 2 -----------------------------
        cross_req_qty_dict = multi_dict(3, float)
        cross_ceil_req_qty_dict = multi_dict(3, float)
        for index, row in network_df.iterrows():
            try:
                sku = row['SKU']
                depot = row['Depot']
                parent_code = row['Parent Code']

                if parent_code == '0':
                    continue

                try:
                    zone = depot_dict[depot]['zone']
                except:
                    zone = "#N/A"
                    depot_dict[depot]['zone'] = "#N/A"
                if zone != "Plant":
                    sit = 0
                    sto = 0
                    total_demand = 0
                    for i in range(stage+1):
                        total_demand += demand_dict[i][sku][depot]
                    try:
                        sit = sit_dict[sku][depot]
                    except Exception as e:
                        print("Error", e)
                        sit = 0
                        sit_dict[sku][depot] = 0
                        sit_map[(sku, depot)] = 0
                    try: 
                        sto = sto_dict[sku][depot]
                    except Exception as e:
                        print("Error", e)
                        sto = 0
                        sto_dict[sku][depot] = 0
                        sto_map[(sku, depot)] = 0
                    if round == 1:
                        cross_allocation_dict[stage][sku][parent_code][depot] = 0
                    allocated = allocation_dict[stage][sku][parent_code][depot] + cross_allocation_dict[stage][sku][parent_code][depot]
                    for i in range(stage):
                        allocated += allocation_dict[i][sku][parent_code][depot] + cross_allocation_dict[i][sku][parent_code][depot]

                    cross_req_qty_dict[sku][parent_code][depot] = max(math.ceil(total_demand*round/rounds) - sit - sto - allocated, 0)
                    cross_ceil_req_qty_dict[sku][parent_code][depot] = math.ceil(cross_req_qty_dict[sku][parent_code][depot]/pallet_dict[sku][depot])*pallet_dict[sku][depot]

                    cross_req_qty_map[stage][(sku, parent_code, depot)] = max(total_demand - sit - sto - allocated + cross_allocation_dict[stage][sku][parent_code][depot], 0)
            except Exception as e:
                cross_req_qty_dict[sku][parent_code][depot] = 0
                print("Error in calculating Req Qty:", e, " at SKU:", sku)

        # ------------ Cross Demand Ratio----------------------------------

        cross_demand_ratio_dict = multi_dict(3, float)
        cross_total_req_qty_dict = multi_dict(2, float)

        cross_demand_ratio_map = {}

        for index, row in network_df.iterrows():
            try:
                sku = row['SKU']
                depot = row['Depot']
                parent_code = row['Parent Code']

                if parent_code == '0':
                    continue

                try:
                    zone = depot_dict[depot]['zone']
                except:
                    zone = "#N/A"
                    depot_dict[depot]['zone'] = "#N/A"
                if zone != "Plant":
                    cross_total_req_qty_dict[sku][parent_code] = sum(cross_req_qty_dict[sku][parent_code].values())
                    if cross_total_req_qty_dict[sku][parent_code] != 0:
                        cross_demand_ratio_dict[sku][parent_code][depot] = cross_req_qty_dict[sku][parent_code][depot] / cross_total_req_qty_dict[sku][parent_code]
                    else:
                        cross_demand_ratio_dict[sku][parent_code][depot] = 0
                cross_demand_ratio_map[(sku, parent_code, depot)] = cross_demand_ratio_dict[sku][parent_code][depot]
            except Exception as e:
                print("Error in calculating PSO Demand Ratio", e, " at SKU:", sku)
                cross_demand_ratio_dict[sku][parent_code][depot] = 0

        # ------------------ Cross Allocation ----------------------------------------------------------------

        cross_leftover_stock_dict_1 = copy.deepcopy(cross_leftover_stock_dict[stage])
        cross_leftover_stock_map_1 = copy.deepcopy(cross_leftover_stock_map[stage])

        for index, row in network_df.iterrows():
            try:
                sku = row['SKU']
                depot = row['Depot']
                parent_code = row['Parent Code']

                if parent_code == '0':
                    continue

                try:
                    zone = depot_dict[depot]['zone']
                except:
                    zone = "#N/A"
                    depot_dict[depot]['zone'] = "#N/A"        
                if zone != "Plant": 
                    alternate_parent = alternate_parent_dict[(sku, depot, parent_code)]

                    if alternate_parent == '0':
                        continue

                    cross_allocation = min(cross_req_qty_dict[sku][parent_code][depot] , cross_leftover_stock_dict_1[sku][alternate_parent]*cross_demand_ratio_dict[sku][parent_code][depot], cross_leftover_stock_dict[stage][sku][alternate_parent])            
                    ceil_allocation = math.ceil(cross_allocation/pallet_dict[sku][depot])*pallet_dict[sku][depot]
                    cross_allocation = math.floor(cross_allocation/pallet_dict[sku][depot])*pallet_dict[sku][depot]

                    if (sum(cross_ceil_req_qty_dict[sku][parent_code].values()) <= cross_leftover_stock_dict_1[sku][alternate_parent] or (cross_allocation == 0 and ceil_allocation <= 2*cross_req_qty_dict[sku][parent_code][depot])) and ceil_allocation <= cross_leftover_stock_dict[stage][sku][alternate_parent]:
                        cross_allocation = ceil_allocation

                    if cross_allocation != 0:
                        stage_allocation_map[stage][(depot, alternate_parent)][category_dict[sku]][sku] += cross_allocation
                        dispatch_dict[(depot, alternate_parent)][category_dict[sku]][sku] = 1

                    try:
                        hub = depot_dict[depot]['hub code']
                        total_tonnage_allocation_dict[depot][alternate_parent] = total_tonnage_allocation_dict[depot][alternate_parent] + (cross_allocation*sku_wt_dict[sku]/1000)
                        tonnage_cross_allocation_dict[stage][depot][alternate_parent] = tonnage_cross_allocation_dict[stage][depot][alternate_parent] + (cross_allocation*sku_wt_dict[sku]/1000)

                        total_hub_tonnage_allocation_dict[hub][alternate_parent] = total_hub_tonnage_allocation_dict[hub][alternate_parent] + (cross_allocation*sku_wt_dict[sku]/1000)
                        hub_tonnage_cross_allocation_dict[stage][hub][alternate_parent] = hub_tonnage_cross_allocation_dict[stage][hub][alternate_parent] + (cross_allocation*sku_wt_dict[sku]/1000)
                    except Exception as e:
                        print("Error", e)
                        total_tonnage_allocation_dict[depot][alternate_parent] = cross_allocation*sku_wt_dict[sku]/1000                        
                        tonnage_cross_allocation_dict[stage][depot][alternate_parent] = cross_allocation*sku_wt_dict[sku]/1000                        
                        total_hub_tonnage_allocation_dict[hub][alternate_parent] = total_hub_tonnage_allocation_dict[hub][alternate_parent] + (cross_allocation*sku_wt_dict[sku]/1000)
                        hub_tonnage_cross_allocation_dict[stage][hub][alternate_parent] = hub_tonnage_cross_allocation_dict[stage][hub][alternate_parent] + (cross_allocation*sku_wt_dict[sku]/1000)

                    cross_leftover_stock_dict[stage][sku][alternate_parent] = cross_leftover_stock_dict[stage][sku][alternate_parent] - cross_allocation 
                    cross_leftover_stock_map[stage][(sku, alternate_parent)] = cross_leftover_stock_dict[stage][sku][alternate_parent]

                    cross_allocation_dict[stage][sku][parent_code][depot] += cross_allocation
                    cross_allocation_map[stage][(sku, parent_code, depot)] = cross_allocation_dict[stage][sku][parent_code][depot]

                    cross_shortage_dict[stage][sku][parent_code][depot] = max(cross_req_qty_dict[sku][parent_code][depot] - cross_allocation,0)
                    cross_shortage_map[stage][(sku, parent_code, depot)] = cross_shortage_dict[stage][sku][parent_code][depot]

            except Exception as e:
                print("Error in calculating Cross Allocation (PSO)", e, " at SKU:", sku, "Plant:", parent_code, "Alternate Plant", alternate_parent)
                cross_allocation = 0
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.3f} seconds")


# In[29]:


start_time = time.time()

allocation_df['SKU Code'] = network_df['SKU']
allocation_df['SKU Desc'] = allocation_df['SKU Code'].map(desc_dict)
allocation_df['Depot'] = network_df['Depot']
allocation_df['Parent Code'] = network_df['Parent Code']
allocation_df['Depot Name'] =allocation_df['Depot'].map(depot_dict_map['name'])
allocation_df['Parent location Name'] = allocation_df['Parent Code'].map(plant_name_dict)
allocation_df['Alternate Plant'] = network_df['Alternate Plant']  
allocation_df['Key'] = allocation_df['SKU Code'].astype(str) + allocation_df['Depot'].astype(str)
allocation_df['SKU Plant Key'] = allocation_df['SKU Code'].astype(str) + allocation_df['Parent Code'].astype(str)
allocation_df['Panipat SKUs'] = np.where((allocation_df['Parent Code'] == '1021'), "Yes", "No")

allocation_df['Category'] = allocation_df['SKU Code'].map(category_dict)
allocation_df['Pack Size'] = allocation_df['SKU Code'].map(pack_dict)
allocation_df['MOQ'] = pd.Series(list(zip(allocation_df['SKU Code'], allocation_df['Depot']))).map(pallet_map)
# allocation_df['MOQ'] = allocation_df['SKU Code'].map(moq_dict)
allocation_df['Zone'] = allocation_df['Depot'].map(depot_dict_map['zone'])

allocation_df['Stock+SIT'] = pd.Series(list(zip(allocation_df['SKU Code'], allocation_df['Depot']))).map(sit_map)
allocation_df['STO'] = pd.Series(list(zip(allocation_df['SKU Code'], allocation_df['Depot']))).map(sto_map)

for i in range(stages):
    allocation_df[f'Total Demand ({stage_dict[i]})'] = pd.Series(list(zip(allocation_df['SKU Code'], allocation_df['Depot']))).map(demand_map[i])

    allocation_df[f'Req Qty ({stage_dict[i]})'] = pd.Series(list(zip(allocation_df['SKU Code'], allocation_df['Parent Code'],allocation_df['Depot']))).map(req_qty_map[i])
    allocation_df[f'Allocation ({stage_dict[i]})'] = pd.Series(list(zip(allocation_df['SKU Code'], allocation_df['Parent Code'],allocation_df['Depot']))).map(allocation_map[i])
    allocation_df[f'Shortage Qty ({stage_dict[i]})'] = pd.Series(list(zip(allocation_df['SKU Code'], allocation_df['Parent Code'],allocation_df['Depot']))).map(shortage_map[i])

    allocation_df[f'Cross Req Qty ({stage_dict[i]})'] = pd.Series(list(zip(allocation_df['SKU Code'], allocation_df['Parent Code'],allocation_df['Depot']))).map(cross_req_qty_map[i])
    allocation_df[f'Cross Allocation ({stage_dict[i]})'] = pd.Series(list(zip(allocation_df['SKU Code'], allocation_df['Parent Code'],allocation_df['Depot']))).map(cross_allocation_map[i])
    allocation_df[f'Final Shortage Qty ({stage_dict[i]})'] = pd.Series(list(zip(allocation_df['SKU Code'], allocation_df['Parent Code'],allocation_df['Depot']))).map(cross_shortage_map[i])

    allocation_df[f'Plant Leftover Stock ({stage_dict[i]})'] = pd.Series(list(zip(allocation_df['SKU Code'], allocation_df['Depot']))).map(leftover_stock_map[i])
    allocation_df[f'Final Plant Leftover Stock ({stage_dict[i]})'] = pd.Series(list(zip(allocation_df['SKU Code'], allocation_df['Depot']))).map(cross_leftover_stock_map[i])

    allocation_df[f'Plant Leftover Stock ({stage_dict[i]})'] = np.where((allocation_df['Zone'] != "Plant"), 0, allocation_df[f'Plant Leftover Stock ({stage_dict[i]})'])
    allocation_df[f'Final Plant Leftover Stock ({stage_dict[i]})'] = np.where((allocation_df['Zone'] != "Plant"), 0, allocation_df[f'Final Plant Leftover Stock ({stage_dict[i]})'])

allocation_df = allocation_df.fillna(0)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.3f} seconds")


# In[30]:


start_time = time.time()


for index, row in allocation_df.iterrows():
    sku = row['SKU Code']
    depot = row['Depot']
    parent_code = row['Parent Code']
    alternate_parent = row['Alternate Plant']
    zone = row['Zone']

    if sit_dict[sku][depot] != row['Stock+SIT']:
        allocation_df.loc[index, 'Stock+SIT'] = sit_map[(sku, depot)]
        allocation_df.loc[index, 'STO'] = sto_dict[sku][depot]

        for i in range(stages):
            if zone != "Plant":
                allocation_df.loc[index, f'Total Demand ({stage_dict[i]})'] = demand_map[i][(sku, depot)]

                allocation_df.loc[index, f'Req Qty ({stage_dict[i]})'] = req_qty_map[i][(sku, parent_code, depot)]
                allocation_df.loc[index, f'Allocation ({stage_dict[i]})'] = allocation_map[i][(sku, parent_code, depot)]
                allocation_df.loc[index, f'Shortage Qty ({stage_dict[i]})'] = shortage_map[i][(sku, parent_code, depot)]

                allocation_df.loc[index, f'Cross Req Qty ({stage_dict[i]})'] = cross_req_qty_map[i][(sku, parent_code, depot)]
                allocation_df.loc[index, f'Cross Allocation ({stage_dict[i]})'] = cross_allocation_map[i][(sku, parent_code, depot)]
                allocation_df.loc[index, f'Final Shortage Qty ({stage_dict[i]})'] = cross_shortage_map[i][(sku, parent_code, depot)]

            if zone == "Plant":         
                allocation_df.loc[index, f'Plant Leftover Stock ({stage_dict[i]})'] = leftover_stock_dict[i][sku][depot]
                allocation_df.loc[index, f'Final Plant Leftover Stock ({stage_dict[i]})'] = cross_leftover_stock_dict[i][sku][depot]

allocation_df = allocation_df.fillna(0)

allocation_df['Final Allocation'] = sum(allocation_df[f'Allocation ({stage_dict[i]})'] for i in range(stages))  # allocation_df['Allocation (PSO)'] + allocation_df['Allocation (Safety Stock)'] + allocation_df['Allocation (M1)'] + allocation_df['Allocation (M2)']
allocation_df['Final Cross Allocation'] = sum(allocation_df[f'Cross Allocation ({stage_dict[i]})'] for i in range(stages)) # allocation_df['Cross Allocation (PSO)'] + allocation_df['Cross Allocation (Safety Stock)'] + allocation_df['Cross Allocation (M1)'] + allocation_df['Cross Allocation (M2)']
allocation_df['Total Allocation'] = allocation_df['Final Cross Allocation'] + allocation_df['Final Allocation']
allocation_df['Final Depot Stock'] = allocation_df['Stock+SIT'] + allocation_df['STO'] + allocation_df['Total Allocation']

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.3f} seconds")


# In[31]:


# start_time = time.time()

# allocation_df.to_csv("Allocation Results.csv", index=False)

# print('DataFrame is written to File successfully.')

# end_time = time.time()
# execution_time = end_time - start_time
# print(f"Execution time: {execution_time:.3f} seconds")


# In[32]:


# Allocation Summary
start_time = time.time()

column_names = [
    'Depot',
    'Plant',
    'SKU Code',
    'Description',
    'Pallet MOQ',
    'PSO',
    'Safety Stock',
    'M1',
    'M2',
    'Total Allocation',
    'Priority',
]

allocation_summary_df = pd.DataFrame(columns = column_names)
temp_regional_df = pd.DataFrame(columns = column_names)
temp_alternate_df = pd.DataFrame(columns = column_names)
temp_same_df = pd.DataFrame(columns = column_names)

filter_df = allocation_df.loc[(allocation_df['Zone'] != "Plant") & (allocation_df['Parent Code'] != allocation_df['Alternate Plant'])]
filter_same_df = allocation_df.loc[(allocation_df['Zone'] != "Plant") & (allocation_df['Parent Code'] == allocation_df['Alternate Plant'])]

# filter_regional_df = filter_df.loc[filter_df['Final Allocation'] > 0]
# filter_cross_df = filter_df.loc[filter_df['Final Cross Allocation'] > 0]

temp_regional_df['Depot'] = filter_df['Depot']
temp_regional_df['Plant'] = filter_df['Parent Code']
temp_regional_df['SKU Code'] = filter_df['SKU Code']
temp_regional_df['Description'] = filter_df['SKU Desc']
temp_regional_df['PSO'] = filter_df['Allocation (PSO)']
temp_regional_df['Safety Stock'] = filter_df['Allocation (Safety Stock)']
temp_regional_df['M1'] = filter_df['Allocation (M1)']
temp_regional_df['M2'] = filter_df['Allocation (M2)']

temp_alternate_df['Depot'] = filter_df['Depot']
temp_alternate_df['Plant'] = filter_df['Alternate Plant']
temp_alternate_df['SKU Code'] = filter_df['SKU Code']
temp_alternate_df['Description'] = filter_df['SKU Desc']
temp_alternate_df['PSO'] = filter_df['Cross Allocation (PSO)']
temp_alternate_df['Safety Stock'] = filter_df['Cross Allocation (Safety Stock)']
temp_alternate_df['M1'] = filter_df['Cross Allocation (M1)']
temp_alternate_df['M2'] = filter_df['Cross Allocation (M2)']

temp_same_df['Depot'] = filter_same_df['Depot']
temp_same_df['Plant'] = filter_same_df['Parent Code']
temp_same_df['SKU Code'] = filter_same_df['SKU Code']
temp_same_df['Description'] = filter_same_df['SKU Desc']
temp_same_df['PSO'] = filter_same_df['Allocation (PSO)'] + filter_same_df['Cross Allocation (PSO)']
temp_same_df['Safety Stock'] = filter_same_df['Allocation (Safety Stock)'] + filter_same_df['Cross Allocation (Safety Stock)']
temp_same_df['M1'] = filter_same_df['Allocation (M1)'] + filter_same_df['Cross Allocation (M1)']
temp_same_df['M2'] = filter_same_df['Allocation (M2)'] + filter_same_df['Cross Allocation (M2)']

allocation_summary_df = pd.concat([temp_regional_df, temp_alternate_df, temp_same_df], ignore_index=True)
allocation_summary_df.fillna(0, inplace=True)
allocation_summary_df['Total Allocation'] = allocation_summary_df['PSO'] + allocation_summary_df['Safety Stock'] + allocation_summary_df['M1'] + allocation_summary_df['M2']
allocation_summary_df['Priority'] = allocation_summary_df['Priority'].astype(str)
allocation_summary_df = allocation_summary_df.loc[(allocation_summary_df['Plant'] != '0')]
allocation_summary_df['Pallet MOQ'] = pd.Series(list(zip(allocation_summary_df['SKU Code'], allocation_summary_df['Depot']))).map(pallet_map)

for index, row in allocation_summary_df.iterrows():
    try:
        sku = row['SKU Code']
        depot = row['Depot']
        try:
            soh = soh_map[(sku, depot)]
        except Exception as e:
            soh = 0
        try:
            demand_total = demand_dict[1][sku][depot] + demand_dict[0][sku][depot]
        except Exception as e:
            demand_total = 0

        if demand_total == 0:
            coverage = 1
        else:
            coverage = soh/demand_total
        if coverage >= 0.67:
            priority = "Green"
        elif coverage >= 0.34:
            priority = "Yellow"
        elif coverage > 0:
            priority = "Red"
        else:
            priority = "Black"

        allocation_summary_df.loc[index, 'Priority'] = priority
    except Exception as e:
        print(e, "Error in Priority", sku, depot)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.3f} seconds")


# In[33]:


# start_time = time.time()

# allocation_summary_df.to_csv("Allocation Summary.csv", index=False)

# print('DataFrame is written to File successfully.')

# end_time = time.time()
# execution_time = end_time - start_time
# print(f"Execution time: {execution_time:.3f} seconds")


# In[34]:


# Depot SKU Level Summary
start_time = time.time()
column_names =  [
    'SKU Code',
    'Description',
    'Depot',
    'Depot Name',
    'Pallet MOQ',
    'Pack Size',
    'SOH',
    'PSO',
    'Safety Stock',
    'M1',
    'M2',
    'Coverage (months)',
    'PSO Allocation',
    'Safety Stock Allocation',
    'M1 Allocation',
    'M2 Allocation',
    'Final Coverage % (before Allocation)',
    'Final Coverage % (after Allocation)',

]
filter_df = allocation_df.copy(deep=True)
filter_df = filter_df.loc[filter_df['Zone']!="Plant"]
filter_df['Total Demand'] = filter_df['Total Demand (PSO)'] + filter_df['Total Demand (Safety Stock)'] + filter_df['Total Demand (M1)'] + filter_df['Total Demand (M2)']
filter_df['SOH'] = filter_df['Stock+SIT'] + filter_df['STO']
filter_df['Cond'] = filter_df['Total Demand'] + filter_df['SOH']
filter_df = filter_df.loc[filter_df['Cond'] > 0]

filter_df['Pack Size'] = filter_df['SKU Code'].map(pack_dict)

depot_sku_summary = pd.DataFrame(columns=column_names)
depot_sku_summary['Depot'] = filter_df['Depot']
depot_sku_summary['SKU Code'] = filter_df['SKU Code']
depot_sku_summary['Description'] = filter_df['SKU Desc']
depot_sku_summary['Pack Size'] = filter_df['Pack Size']
depot_sku_summary['Depot Name'] = filter_df['Depot'].map(depot_dict_map['name'])
depot_sku_summary['SOH'] = filter_df['SOH']
depot_sku_summary['PSO'] = filter_df['Total Demand (PSO)'] 
depot_sku_summary['Safety Stock'] = filter_df['Total Demand (Safety Stock)'] 
depot_sku_summary['M1'] = filter_df['Total Demand (M1)']
depot_sku_summary['M2'] = filter_df['Total Demand (M2)']

depot_sku_summary['PSO Allocation'] = filter_df['Allocation (PSO)'] + filter_df['Cross Allocation (PSO)']
depot_sku_summary['Safety Stock Allocation'] = filter_df['Allocation (Safety Stock)'] + filter_df['Cross Allocation (Safety Stock)']
depot_sku_summary['M1 Allocation'] = filter_df['Allocation (M1)'] + filter_df['Cross Allocation (M1)']
depot_sku_summary['M2 Allocation'] = filter_df['Allocation (M2)'] + filter_df['Cross Allocation (M2)']
depot_sku_summary['Pallet MOQ'] = pd.Series(list(zip(depot_sku_summary['SKU Code'], depot_sku_summary['Depot']))).map(pallet_map)

result = depot_sku_summary[['Pallet MOQ', 'SOH', 'PSO', 'Safety Stock', 'M1', 'M2', 'PSO Allocation', 'Safety Stock Allocation', 'M1 Allocation', 'M2 Allocation']].multiply(depot_sku_summary['Pack Size'], axis="index")
depot_sku_summary['Pallet MOQ'] = result['Pallet MOQ']
depot_sku_summary['SOH'] = result['SOH']
depot_sku_summary['PSO'] = result['PSO']
depot_sku_summary['Safety Stock'] = result['Safety Stock']
depot_sku_summary['M1'] = result['M1']
depot_sku_summary['M2'] = result['M2']
depot_sku_summary['PSO Allocation'] = result['PSO Allocation']
depot_sku_summary['Safety Stock Allocation'] = result['Safety Stock Allocation']
depot_sku_summary['M1 Allocation'] = result['M1 Allocation']
depot_sku_summary['M2 Allocation'] = result['M2 Allocation']

for index, row in depot_sku_summary.iterrows():
    sku = row['SKU Code']
    depot = row['Depot']
    soh = row['SOH']
    pso_allocation = row['PSO Allocation']
    m0_allocation = row['Safety Stock Allocation']
    m1_allocation = row['M1 Allocation']
    m2_allocation = row['M2 Allocation']

    pso = row['PSO']
    m1 = row['Safety Stock']
    m2 = row['M1']
    m3 = row['M2']

    total_demand = pso+m1+m2+m3

    if soh >= total_demand:
        coverage = 3
    elif soh >= m1+m2+pso and m3 > 0:
        coverage = 2 + ((soh-m1-m2-pso)/m3) 
    elif soh >= m1+pso and m2>0:
        coverage = 1 + ((soh-m1-pso)/m2)
    elif soh >= pso and m1>0:
        coverage = (soh-pso)/m1
    else:
        coverage = 0

    depot_sku_summary.loc[index, 'Coverage (months)'] = coverage

filter_df = depot_sku_summary.loc[depot_sku_summary['PSO']+depot_sku_summary['Safety Stock']+depot_sku_summary['M1']+depot_sku_summary['M2'] > 0]
filter_df['Final Coverage % (before Allocation)'] = (filter_df['SOH']) / (filter_df['PSO']+filter_df['Safety Stock']+filter_df['M1']+filter_df['M2'])
filter_df['Final Coverage % (after Allocation)'] = (filter_df['SOH']+filter_df['PSO Allocation']+filter_df['Safety Stock Allocation']+filter_df['M1 Allocation']+filter_df['M2 Allocation']) / (filter_df['PSO']+filter_df['Safety Stock']+filter_df['M1']+filter_df['M2'])

filter_df['Final Coverage % (before Allocation)'] = filter_df['Final Coverage % (before Allocation)'].astype('float64')
filter_df['Final Coverage % (after Allocation)'] = filter_df['Final Coverage % (after Allocation)'].astype('float64')

depot_sku_summary['Final Coverage % (before Allocation)'] = 0.0
depot_sku_summary['Final Coverage % (after Allocation)'] = 0.0

depot_sku_summary.loc[filter_df.index, 'Final Coverage % (before Allocation)'] = filter_df.loc[filter_df.index, 'Final Coverage % (before Allocation)']
depot_sku_summary.loc[filter_df.index, 'Final Coverage % (after Allocation)'] = filter_df.loc[filter_df.index, 'Final Coverage % (after Allocation)']

# depot_sku_summary.drop(columns = ['Pack Size'], inplace = True)
depot_sku_summary.fillna(0, inplace = True)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.3f} seconds")


# In[35]:


# start_time = time.time()

# depot_sku_summary.to_csv("Depot-SKU Summary.csv", index=False)

# print('DataFrame is written to File successfully.')

# end_time = time.time()
# execution_time = end_time - start_time
# print(f"Execution time: {execution_time:.3f} seconds")


# In[36]:


# Plant SKU Level Summary
start_time = time.time()
column_names =  [
    'SKU Code',
    'Description',
    'Plant',
    'Plant Name',
    'SOH',
    'PSO',
    'Safety Stock',
    'M1',
    'M2',
]
filter_df = allocation_df.copy(deep=True)
filter_df = filter_df.loc[filter_df['Zone'] == "Plant"]
filter_df['SOH'] = filter_df['Stock+SIT']

filter_df['Pack Size'] = filter_df['SKU Code'].map(pack_dict)

plant_sku_summary = pd.DataFrame(columns=column_names)
plant_sku_summary['SKU Code'] = filter_df['SKU Code']
plant_sku_summary['Description'] = filter_df['SKU Desc']
plant_sku_summary['Plant'] = filter_df['Depot']
plant_sku_summary['Plant Name'] = plant_sku_summary['Plant'].map(plant_name_dict)

plant_sku_summary['Pack Size'] = filter_df['Pack Size']

plant_sku_summary['SOH'] = filter_df['Stock+SIT']
plant_sku_summary['PSO'] = filter_df['Final Plant Leftover Stock (PSO)']
plant_sku_summary['Safety Stock'] = filter_df['Final Plant Leftover Stock (Safety Stock)']
plant_sku_summary['M1'] = filter_df['Final Plant Leftover Stock (M1)']
plant_sku_summary['M2'] = filter_df['Final Plant Leftover Stock (M2)']

result = plant_sku_summary[['SOH', 'PSO', 'Safety Stock', 'M1', 'M2']].multiply(plant_sku_summary['Pack Size'], axis="index")
plant_sku_summary['SOH'] = result['SOH']
plant_sku_summary['PSO'] = result['PSO']
plant_sku_summary['Safety Stock'] = result['Safety Stock']
plant_sku_summary['M1'] = result['M1']
plant_sku_summary['M2'] = result['M2']

plant_sku_summary.drop(columns = ['Pack Size'], inplace = True)
plant_sku_summary.fillna(0, inplace = True)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.3f} seconds")


# In[37]:


# start_time = time.time()

# plant_sku_summary.to_csv("Plant-SKU Summary.csv", index=False)

# print('DataFrame is written to File successfully.')

# end_time = time.time()
# execution_time = end_time - start_time
# print(f"Execution time: {execution_time:.3f} seconds")


# In[38]:


# Lane Allocation
start_time = time.time()

column_names = [
    'Depot',
    'Hub',
    'Plant',
    'PSO Allocation (Tonne)',
    # 'Cross PSO Allocation (Tonne)',
    'Safety Stock Allocation (Tonne)',
    # 'Cross Safety Stock Allocation (Tonne)',
    'Demand Allocation M1 (Tonne)',
    # 'Cross Demand Allocation M1 (Tonne)',
    'Demand Allocation M2 (Tonne)',
    # 'Cross Demand Allocation M2 (Tonne)',
    'Total Qty Allocated (tonne)',

    'Qty Dispatched (suggestions)',
    'Qty left (suggestion)',
    'Remark (suggestion)',
    'Extra Allocation (suggestion)',
    'Removed Allocation (suggestion)',
    'Total Weight (after incorporating suggestions)'
]


extra_list = []
remove_list = []
remove_total = []
temp_dict_list = {}
idx = 0

for depot in total_tonnage_allocation_dict.keys():
    for plant in total_tonnage_allocation_dict[depot]:
        total_qty_allocated = total_tonnage_allocation_dict[depot][plant]

        pso_allocated = tonnage_allocation_dict[0][depot][plant]
        cross_pso_allocated = tonnage_cross_allocation_dict[0][depot][plant]

        demand_allocated_m1 = tonnage_allocation_dict[1][depot][plant]
        cross_demand_allocated_m1 = tonnage_cross_allocation_dict[1][depot][plant]

        demand_allocated_m2 = tonnage_allocation_dict[2][depot][plant]
        cross_demand_allocated_m2 = tonnage_cross_allocation_dict[2][depot][plant]

        demand_allocated_m3 = tonnage_allocation_dict[3][depot][plant]
        cross_demand_allocated_m3 = tonnage_cross_allocation_dict[3][depot][plant]

        qty_dispatched =  math.floor(total_qty_allocated/minimum_truck_load)*minimum_truck_load
        qty_left = total_qty_allocated - qty_dispatched
        if qty_left >= 13:
            qty_dispatched = qty_dispatched + qty_left
            qty_left = 0
        remark = ""

        if qty_left == 0:
            remark = "Satisfied"
        else:
            for index, row in remarks_data.iterrows():
                if qty_left >= row['Start Weight (tonne)'] and qty_left < row['End Weight (tonne)']:
                    remark = row['Remark']
                    break

        if remark == "Make it FTL":
            extra_list.append((depot, plant, qty_left))
        elif remark == "Wait":
            if qty_dispatched != 0:
                remove_list.append((depot, plant, qty_left))
            else:
                remove_total.append((depot, plant, qty_left))
        else:
            qty_left = 0

        temp_dict = {
            'Depot': depot,
            'Plant': plant,
            'Hub': depot_dict[depot]['hub code'],
            'PSO Allocation (Tonne)': pso_allocated+cross_pso_allocated,
            # 'Cross PSO Allocation (Tonne)': cross_pso_allocated,
            'Safety Stock Allocation (Tonne)': demand_allocated_m1+cross_demand_allocated_m1,
            # 'Cross Safety Stock Allocation (Tonne)': cross_demand_allocated_m1,
            'Demand Allocation M1 (Tonne)': demand_allocated_m2+cross_demand_allocated_m2,
            # 'Cross Demand Allocation M1 (Tonne)': cross_demand_allocated_m2,
            'Demand Allocation M2 (Tonne)': demand_allocated_m3+cross_demand_allocated_m3,
            # 'Cross Demand Allocation M2 (Tonne)': cross_demand_allocated_m3,
            'Total Qty Allocated (tonne)': total_qty_allocated,

            'Qty Dispatched (suggestions)': qty_dispatched,
            'Qty left (suggestion)': qty_left,  
            'Remark (suggestion)': remark,
        }

        temp_dict_list[idx] = temp_dict
        idx = idx + 1        

lane_allocation_df = pd.DataFrame.from_dict(temp_dict_list, orient="index", columns = column_names)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.3f} seconds")


# In[39]:


# start_time = time.time()

# column_names = [
#     'Hub',
#     'Plant',
#     'PSO Allocation (Tonne)',
#     'Cross PSO Allocation (Tonne)',
#     'Demand Allocation (Tonne)',
#     'Cross Demand Allocation (Tonne)',
#     'Total Qty Allocated (tonne)'
# ]
# hub_allocation_df = pd.DataFrame(columns = column_names)

# for hub in total_hub_tonnage_allocation_dict.keys():
#     for plant in total_hub_tonnage_allocation_dict[hub]:
#         total_qty_allocated = total_hub_tonnage_allocation_dict[hub][plant]
#         pso_allocated = hub_tonnage_allocation_dict[0][hub][plant]
#         cross_pso_allocated = hub_tonnage_cross_allocation_dict[0][hub][plant]
#         demand_allocated = hub_tonnage_allocation_dict_m1[hub][plant]
#         cross_demand_allocated = hub_tonnage_cross_allocation_dict_m1[hub][plant]

#         temp_dict = {
#             'Hub': hub,
#             'Plant': plant,
#             'PSO Allocation (Tonne)': pso_allocated,
#             'Cross PSO Allocation (Tonne)': cross_pso_allocated,
#             'Demand Allocation (Tonne)': demand_allocated,
#             'Cross Demand Allocation (Tonne)': cross_demand_allocated,
#             'Total Qty Allocated (tonne)': total_qty_allocated
#         }
#         temp_df = pd.DataFrame([temp_dict])
#         hub_allocation_df = pd.concat([hub_allocation_df, temp_df], ignore_index=True)

# # print(hub_allocation_df)

# end_time = time.time()
# execution_time = end_time - start_time
# print(f"Execution time: {execution_time:.3f} seconds")


# In[40]:


# start_time = time.time()

# hub_allocation_df.to_csv("Hub Allocation.csv", index=False)

# print('DataFrame is written to File successfully.')

# end_time = time.time()
# execution_time = end_time - start_time
# print(f"Execution time: {execution_time:.3f} seconds")


# In[41]:


# Extra Allocation
start_time = time.time()

column_names = [
    'Depot',
    'Plant',
    'SKU',
    'Extra Qty (EA)',
    'Extra Qty (tonnes)',
]
extra_allocation_map = {}
temp_dict_list = {}
idx = 0
final_plant_stock_dict = copy.deepcopy(cross_leftover_stock_dict[3])
for x in extra_list:
    depot = x[0]
    plant = x[1]
    qty_wt = minimum_truck_load - x[2]
    for sku in dispatch_dict[(x[0], x[1])]["A"].keys():
        try:
            try:
                parent_code = parent_dict[(sku, depot)]
            except Exception as e:
                continue
            try:
                sku_wt = sku_wt_dict[sku]
                moq = max(moq_dict[sku],1)
            except:
                sku_wt = 1
                moq = 1
            qty_needed = math.floor(qty_wt*1000/sku_wt)  # Eaches
            try:
                left_stock = final_plant_stock_dict[sku][plant]
            except:
                left_stock = 0
            qty_allowed = min(math.floor(qty_needed/moq)*moq, math.floor(left_stock/moq)*moq)

            try:
                total_qty_allocated = 0
                for stage in range(stages):
                    total_qty_allocated += allocation_dict[stage][sku][parent_code][depot] + cross_allocation_dict[stage][sku][parent_code][depot]
            except Exception as e:
                total_qty_allocated = 0

            try:
                final_depot_stock = sit_dict[sku][depot] + sto_dict[sku][depot] + total_qty_allocated
            except Exception as e:
                final_depot_stock = total_qty_allocated
            try:
                total_demand = 0 
                for stage in range(stages):
                    total_demand += demand_dict[stage][sku][depot]
            except Exception as e:
                total_demand = 0        
            acceptable_qty = 0

            if final_depot_stock < total_demand*1.2:
                acceptable_qty = total_demand*1.2 - final_depot_stock
                acceptable_qty = math.floor(acceptable_qty / moq)*moq
            # print(sku, depot, plant, left_stock, total_qty_allocated, moq_dict[sku], sku_wt, math.ceil(qty_left*1000/sku_wt))
            extra_qty = min(acceptable_qty, qty_allowed)
            extra_qty_wt = extra_qty*sku_wt/1000
            qty_wt = qty_wt - extra_qty_wt
            if extra_qty > 0:
                temp_dict = {
                    'Depot': depot,
                    'Plant': plant,
                    'SKU': sku,
                    'Extra Qty (EA)': extra_qty,
                    'Extra Qty (tonnes)': extra_qty_wt, 
                    # 'Final Depot Stock': final_depot_stock,
                    # 'Leftover Plant Stock': cross_leftover_stock_dict[3][sku][plant],  
                    # 'Total Demand': total_demand,
                }
                temp_dict_list[idx] = temp_dict
                idx = idx + 1
                final_plant_stock_dict[sku][plant] = final_plant_stock_dict[sku][plant] - extra_qty

                try:
                    extra_allocation_map[(depot, plant)] = extra_allocation_map[(depot, plant)] + extra_qty_wt
                except:
                    extra_allocation_map[(depot, plant)] = extra_qty_wt

            if qty_wt <= 0.05:
                break
        except Exception as e:
            print(e, sku, "-")

extra_allocation_df = pd.DataFrame.from_dict(temp_dict_list, orient="index", columns= column_names)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.3f} seconds")


# In[42]:


# start_time = time.time()

# extra_allocation_df.to_csv("Extra Allocation.csv", index=False)

# print('DataFrame is written to File successfully.')

# end_time = time.time()
# execution_time = end_time - start_time
# print(f"Execution time: {execution_time:.3f} seconds")


# In[43]:


# Removed Allocation
start_time = time.time()

column_names = [
    'Depot',
    'Plant',
    'SKU',
    'Category',
    'Removed Qty (EA)',
    'Removed Qty (tonnes)',
    # 'Depot Stock',
]
removed_allocation_map = {}
total_removed_qty = multi_dict(2, float)
total_removed_wt = multi_dict(2, float)

for x in remove_list:
    depot = x[0]
    plant = x[1]
    qty_wt = x[2] # in tonnes
    depot_plant = (depot, plant)
    error = 0.005
    categories = ["C", "B"]

    for stage in range(stages-1, -1, -1):
        for c in categories:
            if qty_wt <= error:
                break
            for sku in stage_allocation_map[stage][depot_plant][c].keys():
                if qty_wt <= error:
                    break

                sku_allocation = stage_allocation_map[stage][depot_plant][c][sku]
                sku_wt = sku_wt_dict[sku]
                moq = moq_dict[sku]
                eaches = qty_wt*1000/sku_wt
                qty_needed = math.floor(eaches/moq)*moq
                removed_qty = min(qty_needed, sku_allocation)
                removed_wt = removed_qty*sku_wt/1000
                qty_wt = qty_wt - removed_wt

                if removed_qty > 0 and qty_wt >= 0:
                    try:
                        total_removed_qty[(depot, plant)][sku] = total_removed_qty[(depot, plant)][sku] + removed_qty
                        total_removed_wt[(depot, plant)][sku] = total_removed_wt[(depot, plant)][sku] + removed_wt
                    except:
                        total_removed_qty[(depot, plant)][sku] = removed_qty
                        total_removed_wt[(depot, plant)][sku] = removed_wt


for x in remove_total:
    depot = x[0]
    plant = x[1]
    qty_wt = x[2] # in tonnes
    depot_plant = (depot, plant)
    error = 0.001
    categories = ["C", "B","A"]

    for stage in range(stages-1, -1, -1):
        for c in categories:
            if qty_wt <= error:
                break
            for sku in stage_allocation_map[stage][depot_plant][c].keys():
                if qty_wt <= error:
                    break

                sku_allocation = stage_allocation_map[stage][depot_plant][c][sku]
                sku_wt = sku_wt_dict[sku]
                removed_qty = sku_allocation
                removed_wt = removed_qty*sku_wt/1000
                qty_wt = qty_wt - removed_wt

                if removed_qty > 0:
                    try:
                        total_removed_qty[(depot, plant)][sku] = total_removed_qty[(depot, plant)][sku] + removed_qty
                        total_removed_wt[(depot, plant)][sku] = total_removed_wt[(depot, plant)][sku] + removed_wt
                    except:
                        total_removed_qty[(depot, plant)][sku] = removed_qty
                        total_removed_wt[(depot, plant)][sku] = removed_wt


temp_dict_list = {}
idx = 0
for x in total_removed_qty.keys():
    depot = x[0]
    plant = x[1]
    weight = sum(total_removed_wt[x].values())
    removed_allocation_map[x] = weight
    for sku in total_removed_qty[x].keys():
        c = category_dict[sku]
        removed_qty = total_removed_qty[x][sku]
        removed_wt = total_removed_wt[x][sku]
        temp_dict = {
            'Depot': depot,
            'Plant': plant,
            'SKU': sku,
            'Category': c,
            'Removed Qty (EA)': removed_qty,
            'Removed Qty (tonnes)': removed_wt, 
        }
        temp_dict_list[idx] = temp_dict
        idx = idx + 1

remove_allocation_df = pd.DataFrame.from_dict(temp_dict_list, orient="index", columns = column_names)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.3f} seconds")


# In[44]:


# start_time = time.time()

# remove_allocation_df.to_csv("Remove Allocation.csv", index=False)

# print('DataFrame is written to File successfully.')

# end_time = time.time()
# execution_time = end_time - start_time
# print(f"Execution time: {execution_time:.3f} seconds")


# In[45]:


start_time = time.time()

lane_allocation_df['Extra Allocation (suggestion)'] = pd.Series(list(zip(lane_allocation_df['Depot'], lane_allocation_df['Plant']))).map(extra_allocation_map)
lane_allocation_df['Removed Allocation (suggestion)'] = pd.Series(list(zip(lane_allocation_df['Depot'], lane_allocation_df['Plant']))).map(removed_allocation_map)
lane_allocation_df = lane_allocation_df.fillna(0)

lane_allocation_df['Total Weight (after incorporating suggestions)'] = lane_allocation_df['Extra Allocation (suggestion)'] - lane_allocation_df['Removed Allocation (suggestion)'] + lane_allocation_df['Qty left (suggestion)']

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.3f} seconds")


# In[46]:


# start_time = time.time()

# lane_allocation_df.to_csv("Lane Allocation.csv", index=False)

# print('DataFrame is written to File successfully.')

# end_time = time.time()
# execution_time = end_time - start_time
# print(f"Execution time: {execution_time:.3f} seconds")


# In[47]:


start_time = time.time()

with pd.ExcelWriter("Data Allocation Output.xlsx",engine="openpyxl", mode='a', if_sheet_exists='replace') as writer:
    allocation_df.to_excel(writer, sheet_name = "Allocation (EA)", index=False)
    allocation_summary_df.to_excel(writer, sheet_name="Allocation Summary (EA)", index=False)
    depot_sku_summary.to_excel(writer, sheet_name = "Depot-SKU Summary (L)", index=False)
    plant_sku_summary.to_excel(writer, sheet_name = "Plant-SKU Summary (L)", index=False)

    lane_allocation_df.to_excel(writer, sheet_name="Lane Summary (tonne)", index=False)

    extra_allocation_df.to_excel(writer, sheet_name="Extra Allocation (EA)", index=False)
    remove_allocation_df.to_excel(writer, sheet_name="Remove Allocation (EA)", index=False)
    # hub_allocation_df.to_excel(writer, sheet_name="Hub Allocation", index = False)
print('DataFrame is written to File successfully.')

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.3f} seconds")


# In[48]:


finish_time = time.time()
execution_time = (finish_time - initial_time)/60
print("Program Executed Successfully\n", f"Execution time: {execution_time:.3f} minutes")

