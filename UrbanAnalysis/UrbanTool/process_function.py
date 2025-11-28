from UrbanTool.city import City
import pickle

def estimate_models(zone_folder):
	print(f"starting {zone_folder.split('/')[-1]} ......")

	X_common = ['n_nodes', 'm_street', 'bicycle_det', 'car_det', 'bus_det', 'motorcycle_det', 'truck_det',
        'parking meter_det', 'bench_det', 'residential_lu', 'grass_lu', 'forest_lu', 'cemetery_lu'] #'orchard_lu'
	X_agg = ['n_nodes', 'm_street', 'bicycle_det', 'car_det', 'bus_det', 'motorcycle_det', 'truck_det', 'parking meter_det', 'bench_det',
        'food_place_amn_agg','education_amn_agg','transportation_amn_agg','financial_amn_agg','entertainment_amn_agg','public_service_amn_agg',
        'facility_amn_agg','waste_amn_agg','other_amn_agg', 'residential_lu', 'grass_lu', 'forest_lu', 'cemetery_lu'] #'orchard_lu'
		
	city = City(zone_folder, True, False, True, cells = True)
	try:
		city.estimate_ols_model('person_det',X_common, agg = 0)
		city.estimate_ols_model('person_det',X_agg, agg = 1)
		print(f"...... {zone_folder.split('/')[-1]} done")

		with open(f'{zone_folder}/file.pkl', 'wb') as file:
			pickle.dump(city, file)
		return 1
	except:
		print(f"...... {zone_folder.split('/')[-1]} failed")
		return -1

def estimate_models2(zone_folder):
	print(f"starting {zone_folder.split('/')[-1]} ......")

	X_common = ['n_nodes', 'm_street', 'bicycle_det', 'car_det', 'bus_det', 'motorcycle_det', 'truck_det',
        'parking meter_det', 'bench_det', 'residential_lu', 'grass_lu', 'forest_lu', 'cemetery_lu'] #'orchard_lu'
	X_agg = ['n_nodes', 'm_street', 'bicycle_det', 'car_det', 'bus_det', 'motorcycle_det', 'truck_det', 'parking meter_det', 'bench_det',
        'food_place_amn_agg','education_amn_agg','transportation_amn_agg','financial_amn_agg','entertainment_amn_agg','public_service_amn_agg',
        'facility_amn_agg','waste_amn_agg','other_amn_agg', 'residential_lu', 'grass_lu', 'forest_lu', 'cemetery_lu'] #'orchard_lu'
		
	city = City(zone_folder, True, False, True, cells = True)
	#try:
	city.estimate_ols_model('person_det',X_common, kind = 'nrm')
	city.estimate_ols_model('person_det',X_agg, kind = 'agg')
	city.estimate_ols_model('person_det',X_agg, kind = 'std')
	print(f"...... {zone_folder.split('/')[-1]} done")

	with open(f'{zone_folder}/obj.pkl', 'wb') as file:
		pickle.dump(city, file)
	return 1
	#except:
	#	print(f"...... {zone_folder.split('/')[-1]} failed")
#		return -1

    