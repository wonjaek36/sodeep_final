from main import Main
import os
import sys
import json

config = None
with open(os.path.join('src', 'config.json'), 'r') as f:
    config = json.load(f)

test_model = config["TEST_MODEL"]
test_set = config["TEST"]

for test in test_set:

	if test_model == "VGGNet":

		print ("test_model" + test_model)
		learning_rate 		= test["learning_rate"]
		batch_size 			= test["batch_size"]
		epoch 				= test["epoch"]
		regularization1 	= test.get("regularization1", None)
		regularization2 	= test.get("regularization2", None)
		regularization3 	= test.get("regularization3", None)
		regularization4 	= test.get("regularization4", None)
		regularization5 	= test.get("regularization5", None)
		regularization6 	= test.get("regularization6", None)
		regularization7 	= test.get("regularization7", None)
		regularization8 	= test.get("regularization8", None)
		dropout1 			= test.get("dropout1", None)
		dropout2 			= test.get("dropout2", None)
		dropout3 			= test.get("dropout3", None)
		name				= test.get("name", None)

		options = {
			"learning_rate" 		:learning_rate 	,
			"batch_size" 			:batch_size 	,
			"epoch" 				:epoch 			,
			"regularization1" 		:regularization1,
			"regularization2" 		:regularization2,
			"regularization3" 		:regularization3,
			"regularization4" 		:regularization4,
			"regularization5" 		:regularization5,
			"regularization6" 		:regularization6,
			"regularization7" 		:regularization7,
			"regularization8" 		:regularization8,
			"dropout1" 				:dropout1 		,
			"dropout2" 				:dropout2 		,
			"dropout3" 				:dropout3 		,
			"name"					:name
		}

		if name is None:
			continue

	if test_model == "MobileNet" or test_model == 'ResNet' or test_model == 'InceptionV3':

		print ("test_model: ", test_model)
		learning_rate 		= test["learning_rate"]
		batch_size 			= test["batch_size"]
		epoch 				= test["epoch"]
		regularization1 	= test.get("regularization1", None)
		regularization2 	= test.get("regularization2", None)
		regularization3 	= test.get("regularization3", None)
		regularization4 	= test.get("regularization4", None)
		regularization5 	= test.get("regularization5", None)
		regularization6 	= test.get("regularization6", None)
		regularization7 	= test.get("regularization7", None)
		regularization8 	= test.get("regularization8", None)
		regularization9 	= test.get("regularization9", None)
		regularization10 	= test.get("regularization10", None)
		regularization11 	= test.get("regularization11", None)
		regularization12 	= test.get("regularization12", None)
		regularization13 	= test.get("regularization13", None)
		regularization14 	= test.get("regularization14", None)
		name				= test.get("name", None)

		options = {
			"learning_rate" 		:learning_rate 	,
			"batch_size" 			:batch_size 	,
			"epoch" 				:epoch 			,
			"regularization1" 		:regularization1,
			"regularization2" 		:regularization2,
			"regularization3" 		:regularization3,
			"regularization4" 		:regularization4,
			"regularization5" 		:regularization5,
			"regularization6" 		:regularization6,
			"regularization7" 		:regularization7,
			"regularization8" 		:regularization8,
			"regularization9" 		:regularization9,
			"regularization10" 		:regularization10,
			"regularization11" 		:regularization11,
			"regularization12" 		:regularization12,
			"regularization13" 		:regularization13,
			"regularization14" 		:regularization14,
			"name"					:name
		}



	main = Main(config)
	main.parse_data()
	main.train(options)
