use_cuda = true
require 'torch'
require 'nn'
if (use_cuda) then
	require 'cutorch'
	require 'cunn'
end
require 'nnx'
require 'optim'
require 'rnn'
require 'model'
require 'data'
require 'image'

use_sgd = true

model_sogou_lstm()
-- model_weibo()
c = use_cuda == true and nn.CTCCriterion():cuda() or nn.CTCCriterion()

-- Prepare the data
load_training_data()
load_test_data()

function recognize(img)

	local inputTable = getInputTableFromImg({img})
	local outputTable = s:forward(inputTable)
	local input_size = table.getn(inputTable)
	local pred = use_cuda and torch.CudaTensor(1, input_size, klass) or torch.Tensor(1, input_size, klass)
	for i = 1, table.getn(inputTable) do
		pred[1][i] = torch.reshape(outputTable[i], 1, klass)
	end

	local pred_str_ary = { }
	local pred_idx = 1
	local last_c = ""
	for i = 1, table.getn(inputTable) do
		local temp, idx = torch.max(pred[1][i], 1)
		pred[1][i][idx[1] ] = -1e10
		if (idx[1] ~= 1) then
			if (last_c ~= label_set[idx[1] - 1]) then
				pred_str_ary[pred_idx] = label_set[idx[1] - 1]
				pred_idx = pred_idx + 1
				last_c = label_set[idx[1] - 1]
			end
		else
			last_c = ""
		end
	end
	local pred_str = table.concat(pred_str_ary)
	return pred_str
end
	
function showDataResult(img_idx, rank_num)
	local img = imgs_type[img_idx]
	local label = labels_type[img_idx]

	local inputTable = getInputTableFromImg({img})

	local outputTable = s:forward(inputTable)
	local input_size = table.getn(inputTable)
	local pred = use_cuda and torch.CudaTensor(1, input_size, klass) or torch.Tensor(1, input_size, klass)
	for i = 1, table.getn(inputTable) do
		pred[1][i] = torch.reshape(outputTable[i], 1, klass)
	end
	label_str = ""
	for i = 1, table.getn(label) do
		label_str = label_str .. label_set[label[i]]
	end
	print("Correct Label: " .. label_str)

	rank_num = rank_num or 3
	rank_num = math.min(rank_num, 5)
	pred_data = { }
	for r = 1,rank_num do
		local pred_str = "               "
		for i = 1, table.getn(inputTable) do
			local temp, idx = torch.max(pred[1][i], 1)
			pred[1][i][idx[1]] = -1e10
			if (idx[1] == 1) then
				pred_str = pred_str .. " "
				if (r == 1) then
					pred_data[i] = -1
				end
			else
				pred_str = pred_str .. label_set[idx[1] - 1]
				if (r == 1) then
					pred_data[i] = idx[1] - 1
				end
			end
		end
		print(pred_str)
	end

end

function showTestResult(img_idx, rank_num)
	imgs_type = imgs_test
	labels_type = labels_test
	return showDataResult(img_idx, rank_num)
end

function showTrainResult(img_idx, rank_num)
	imgs_type = imgs_train
	labels_type = labels_train
	return showDataResult(img_idx, rank_num)
end

function calDataErrRate()
	print("Error rate on " .. type_str .. " set (image number: " .. table.getn(imgs_type) .. ")")
	local err_num = 0
	for img_idx = 1,table.getn(imgs_type) do
		local img = imgs_type[img_idx]
		local label = labels_type[img_idx]

		local inputTable = getInputTableFromImg({img})
		local outputTable = s:forward(inputTable)
		local input_size = table.getn(inputTable)
		local pred = use_cuda and torch.CudaTensor(1, input_size, klass) or torch.Tensor(1, input_size, klass)
		for i = 1, table.getn(inputTable) do
			pred[1][i] = torch.reshape(outputTable[i], 1, klass)
		end

		label_str = ""
		for i = 1, table.getn(label) do
			label_str = label_str .. label_set[label[i]]
		end

		local pred_str_ary = { }
		local pred_idx = 1
		local last_c = ""
		for i = 1, table.getn(inputTable) do
			local temp, idx = torch.max(pred[1][i], 1)
			pred[1][i][idx[1]] = -1e10
			if (idx[1] ~= 1) then
				if (last_c ~= label_set[idx[1] - 1]) then
					pred_str_ary[pred_idx] = label_set[idx[1] - 1]
					pred_idx = pred_idx + 1
					last_c = label_set[idx[1] - 1]
				end
			else
				last_c = ""
			end
		end
		local pred_str = table.concat(pred_str_ary)
		if (pred_str ~= label_str) then
			err_num = err_num + 1
		end
	end
	print("Error rate on " .. type_str .. " set: " .. err_num / table.getn(imgs_type) .. ". " .. err_num .. "/" .. table.getn(imgs_type))
end

function calTestErrRate()
	imgs_type = imgs_test
	labels_type = labels_test
	label_pathname_ary_type = label_pathname_ary_test
	type_str = "test"
	return calDataErrRate()
end

function calTrainErrRate()
	imgs_type = imgs_train
	labels_type = labels_train
	label_pathname_ary_type = label_pathname_ary_train
	type_str = "training"
	return calDataErrRate()
end

x, dl_dx = s:getParameters()

feval = function(x_new)
	if x ~= x_new then
		x:copy(x_new)
	end
	-- inputTable, target = toySample()
	inputTable, target = nextSample()

	dl_dx:zero()
	-- forward of model
	outputTable = s:forward(inputTable)
	-- change the format of output of the nn.Sequencer to match the format of input of CTCCriterion
	local input_size = table.getn(inputTable)
	pred = use_cuda and torch.CudaTensor(batch_size, input_size, klass) or torch.Tensor(batch_size, input_size, klass)
	for b = 1, batch_size do
		for i = 1, table.getn(inputTable) do
			pred[b][i] = torch.reshape(outputTable[i][b], 1, klass)
		end
	end
	-- forward and backward of criterion
	loss_x = c:forward(pred, target)
	gradCTC = c:backward(pred, target)
	-- change the format of gradInput of the CTCCriterion to match the format of output of nn.Sequencer
	gradOutputTable = { }
	for i = 1, table.getn(inputTable) do
		gradOutputTable[i] = use_cuda == true and torch.CudaTensor(batch_size, klass) or torch.Tensor(batch_size, klass)
		for b = 1, batch_size do
			gradOutputTable[i][b] = torch.reshape(gradCTC[b][i], klass)
		end
	end
	s:backward(inputTable, gradOutputTable)
	return loss_x, dl_dx
end

-- sgd parameters
sgd_params = {
	learningRate = 2e-4,
	learningRateDecay = 0,
	weightDecay = 0,
	momentum = 0.9
}

-- adadelta parameters
adadelta_params = {
	rho = 0.95,
	eps = 1e-6
}
state = { }

loss_ary = { }
test_err_rate = { }
train_err_rate = { }
loss_epoch = { }

evalCounter = 0
epoch = 0
function train(batch_num)
	last_epoch = epoch or 0
	star_num = star_num or 0
	line_star = 80
	io.write("Epoch " .. epoch .. ": ")
	star_num = 0
	for i =1,batch_num do
		evalCounter = evalCounter + 1
		if (use_sgd == true) then
			_, fs = optim.sgd(feval, x, sgd_params)
		else
			_, fs = optim.adadelta(feval, x, adadelta_params, state)
		end
		if (i % 1 == 0) then
			local percent = math.floor(evalCounter % batch_num / batch_num * 100)
			if (evalCounter % batch_num == 0) then
				percent = 100
			end
			local cur_star_num = math.floor(percent / (100 / line_star))
			for j = star_num + 1,cur_star_num do
				io.write("=")
			end
			star_num = cur_star_num
			io.flush()
		end
		loss_ary[evalCounter] = fs[1]
	end
end

function train_epoch(epoch_num, batch_size_param)
	batch_size = batch_size_param or 1
	local batch_num = math.floor(table.getn(imgs_train) / batch_size)
	for e = 1, epoch_num do
		nClock = os.clock() 
		epoch = epoch + 1
		train(batch_num)

		shuffle(train_idx_ary)
		train_idx = 1

		local elapse = torch.round((os.clock() - nClock) * 10) / 10

		for j = star_num + 1, line_star do
			io.write("=")
		end
		io.flush()
		local loss_tensor = torch.Tensor(loss_ary)
		local loss_cur_epoch = loss_tensor:sub((epoch - 1) * batch_num + 1, epoch * batch_num):mean()
		io.write(". Ave loss: " .. loss_cur_epoch .. ".")
		loss_epoch[epoch] = loss_cur_epoch
		io.write(" Execution time: " .. elapse .. "s.")
		io.write("\n")

		-- save the model file
		if (epoch % 5 == 1) then
			torch.save("models/" .. epoch .. ".mdl", m)
		end
		calTestErrRate()
	end
end

function load_model(model_idx)
	m = torch.load("models/" .. model_idx .. ".mdl")
	s = use_cuda == true and nn.Sequencer(m):cuda() or nn.Sequencer(m)
	x, dl_dx = s:getParameters()
end

-- load_model(5)
train_epoch(500, 64)
