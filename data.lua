local cv = require 'cv'
require 'cv.imgproc'
require 'cv.imgcodecs'
require 'cv.highgui'
require 'torch'
require 'image'
require 'gnuplot'
require 'lfs'
require 'util'

data_src = "sogou"

function get_label_by_str(label_str)
	char_idx = 1
	local result = { }
	for c in label_str:gmatch(".") do
		for i = 1,table.getn(label_set) do
			if (label_set[i] == c) then
				result[char_idx] = i
			end
		end
		char_idx = char_idx + 1
	end
	return result
end

function load_data()
	local type_idx = 1
	for img_filename in lfs.dir(data_src .. "/" .. type_str .. "_set") do
		if (img_filename ~= "." and img_filename ~= "..") then
			local prefix = mysplit(img_filename, ".")[1]
			local img_filepath = data_src .. "/" .. type_str .. "_set/" .. img_filename
			local label = mysplit(img_filename, ".")[1]

			print(img_filepath)
			local raw_img = cv.imread { img_filepath, cv.IMREAD_COLOR }

			imgs_type[type_idx] = raw_img
			labels_str_type[type_idx] = string.lower(label)
			labels_type[type_idx] = get_label_by_str(string.lower(label))
			type_idx = type_idx + 1
		end
	end


	for i = 1,table.getn(imgs_type) do
		type_idx_ary[i] = i
	end

	print("Finish loading " .. type_str .. " data set. (data size: " .. table.getn(imgs_type) .. ")")
end

function load_training_data()
	imgs_train = { }
	labels_train = { }
	labels_str_train = { }
	train_idx_ary = { }

	imgs_type = imgs_train
	labels_type = labels_train
	labels_str_type = labels_str_train
	type_idx_ary = train_idx_ary
	type_str = "training"

	load_data()

	train_idx = 1
end

function load_test_data()
	imgs_test = { }
	labels_test = { }
	labels_str_test = { }
	test_idx_ary = { }

	imgs_type = imgs_test
	labels_type = labels_test
	labels_str_type = labels_str_test
	type_idx_ary = test_idx_ary
	type_str = "test"

	load_data()

	test_idx = 1
end

function extractFeature(imgs)
        local img_num = table.getn(imgs)
	local size = imgs[1]:size()
	local height = size[1]
	local width = size[2]
	local channel = size[3]
	local mean = 123

	local featureTable = { }
	for i = 1, width do
		local feature = torch.Tensor(img_num, feature_len):fill(0)
		feature = use_cuda and feature:cuda() or feature
		for n = 1, img_num do
			for j = 1, height do
				for c = 1, channel do
       		                 	feature[n][(j - 1) * channel + c] = imgs[n][j][i][c]
				end
			end
		end
		featureTable[i] = (feature - mean) / 255
	end
	return featureTable
end

function getInputTableFromImg(imgs)
	if (use_rnn) then
		return extractFeature(imgs)
	end
        local img_num = table.getn(imgs)
        local size = imgs[1]:size()
        local channel = size[1]
        local height = size[2]
        local width = size[3]

        local inputTable = { }
        for i = 1, width - window + 1 do
                local sub_imgs = use_cuda and torch.CudaTensor(img_num, channel, height, window) or torch.FloatTensor(img_num, channel, height, window)
                for n = 1, img_num do
                        sub_imgs[n] = imgs[n]:sub(1, channel, 1, height, i, i + window - 1)
                end
                inputTable[i] = sub_imgs
        end
        return inputTable
end

function nextSample()
        local labels = { }
        local imgs = { }
        for b = 1, batch_size do
                imgs[b] = imgs_train[train_idx_ary[train_idx] ]
                labels[b] = labels_train[train_idx_ary[train_idx] ]

                train_idx = (train_idx == table.getn(imgs_train)) and 1 or (train_idx + 1)
        end
        local inputTable = getInputTableFromImg(imgs)
        return inputTable, labels
end

function get_label_from_pred(pred)
	local pred_label = { }
	local size = pred:size()
	local length = size[2]
	local klass = size[3]
	for i = 1, length do
		local max = 0
		for k = 1, klass do
			if (pred[1][i][k] > max) then
				pred_label[i] = k
				max = pred[1][i][k]
			end
		end
	end
	return pred_label
end
