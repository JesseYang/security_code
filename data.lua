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

function train_set_pca(w, dim, whiten)
        -- 1. get table of feature vectors
        local features = { }
        local height = imgs_train[1]:size(1)
	local channel = imgs_train[1]:size(3)
        w = w or window
        local idx = 1
        local ori_dim = height * w * channel
        for i = 1, table.getn(imgs_train) do
                local width = imgs_train[i]:size(2)
                for c = 1, width - w + 1 do
                        feature = imgs_train[i]:sub(1, height, c, c + w - 1, 1, channel):reshape(1, ori_dim)
                        features[idx] = feature
                        idx = idx + 1
                end
        end
        local features_tensor = torch.zeros(table.getn(features), ori_dim)
        for i = 1, table.getn(features) do
                features_tensor[i] = features[i][1]
        end

        -- 2. preprocess for the features
        dim = dim or pca_dim
        whiten = whiten or false
        pca_transform = pca(features_tensor, dim, whiten)

        return pca_transform
end

function pca(d, dim, whiten)
        mean_over_dim = torch.mean(d, 1)
        d_m = d - torch.ger(torch.ones(d:size(1)), mean_over_dim:squeeze())
        cov = d_m:t() * d_m
        ce, cvv = torch.symeig(cov, 'V')
        -- sort eigenvalues
        ce, idx = torch.sort(ce, true)
        -- sort eigenvectors
        cvv = cvv:index(2, idx:long())

        print(ce:sub(1, dim):sum() / ce:sum())
        t = cvv:sub(1, -1, 1, dim)
        ce = ce:sub(1, dim)
        pca_data = d_m * t
        whiten_factor = torch.diag(ce:clone():sqrt():pow(-1))
        v1 = torch.var(pca_data:sub(1, -1, 1, 1))
        whiten_factor = whiten_factor * torch.sqrt(1/(v1 * whiten_factor[1][1]^2))
        sigma = torch.sqrt(torch.var(pca_data))
        if (whiten == true) then
                return t * whiten_factor
        else
                return t / sigma
        end
end

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
			imgs_pathname_ary_type[type_idx] = img_filename
			type_idx = type_idx + 1
		end
		if (type_idx == 13) then
			break
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
	imgs_pathname_ary_train = { }

	imgs_type = imgs_train
	labels_type = labels_train
	labels_str_type = labels_str_train
	type_idx_ary = train_idx_ary
	imgs_pathname_ary_type = imgs_pathname_ary_train
	type_str = "training"

	load_data()

	train_idx = 1
end

function load_test_data()
	imgs_test = { }
	labels_test = { }
	labels_str_test = { }
	test_idx_ary = { }
	imgs_pathname_ary_test = { }

	imgs_type = imgs_test
	labels_type = labels_test
	labels_str_type = labels_str_test
	type_idx_ary = test_idx_ary
	imgs_pathname_ary_type = imgs_pathname_ary_test
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
	for i = 1, width - window + 1 do
		local feature = torch.zeros(img_num, feature_len)

		for n = 1, img_num do
			feature[n] = imgs[n]:sub(1, height, i, i + window - 1, 1, channel):reshape(1, feature_len)
		end

		if (use_pca == true) then
			for n = 1, img_num do
				feature[n] = feature[n] - mean_over_dim
			end
			featureTable[i] = feature * pca_transform
		else
			feature = (feature / 255) - 0.5
			featureTable[i] = feature
		end
		featureTable[i] = use_cuda and featureTable[i]:cuda() or featureTable[i]
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
