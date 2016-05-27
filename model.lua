require 'torch'
require 'nn'
require 'nnx'
require 'rnn'

function model_sogou()
	label_set = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z" }
	klass = table.getn(label_set) + 1
	ksize = 5
	window = 32
	height = 44

	m = nn.Sequential()
	-- zero stage: padding to top and bottom to make the height 48
	-- m:add(nn.Padding(2, 2))		-- the first dimension is channel, the second dimension is height
	-- m:add(nn.Padding(2, -2))
	m:add(nn.Narrow(2, 2, 40))
	-- m:add(nn.Normalize(2))
	-- first stage: before: 32 * 40; after 16 * 20
	m:add(nn.SpatialConvolution(3, 16, ksize, ksize, 1, 1, (ksize - 1) / 2, (ksize - 1) / 2))
	m:add(nn.ReLU())
	m:add(nn.SpatialMaxPooling(2, 2, 2, 2))
	-- second stage: before: 16 * 20; after 8 * 10
	m:add(nn.SpatialConvolution(16, 32, ksize, ksize, 1, 1, (ksize - 1) / 2, (ksize - 1) / 2))
	m:add(nn.ReLU())
	m:add(nn.SpatialMaxPooling(2, 2, 2, 2))
	-- third stage: before: 8 * 10; after 4 * 5
	m:add(nn.SpatialConvolution(32, 64, ksize, ksize, 1, 1, (ksize - 1) / 2, (ksize - 1) / 2))
	m:add(nn.ReLU())
	m:add(nn.SpatialMaxPooling(2, 2, 2, 2))
	-- last stage: standard 1-layer mlp
	m:add(nn.Reshape(64 * 4 * 5))
	m:add(nn.Linear(64 * 4 * 5, klass))

	s = use_cuda == true and nn.Sequencer(m):cuda() or nn.Sequencer(m)
end

function model_sogou_lstm()
	-- the rnn model
	use_rnn = true
	use_pca = true
	window = 1
	label_set = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z" }
	klass = table.getn(label_set) + 1
	height = 44
	channel = 3
	feature_len = height * channel
	pca_dim = height * channel
	hidden_size = 200

	l1_1 = nn.LSTM(feature_len, hidden_size)
	l1_2 = nn.LSTM(feature_len, hidden_size)

	fwdSeq_1 = nn.Sequencer(l1_1)
	bwdSeq_1 = nn.Sequencer(l1_2)
	merge_1 = nn.JoinTable(1, 1)
	mergeSeq_1 = nn.Sequencer(merge_1)

	concat_1 = nn.ConcatTable()
	concat_1:add(fwdSeq_1):add(nn.Sequential():add(nn.ReverseTable()):add(bwdSeq_1):add(nn.ReverseTable()))
	brnn_1 = nn.Sequential()
		:add(concat_1)
		:add(nn.ZipTable())
		:add(mergeSeq_1)

	l2_1 = nn.LSTM(2 * hidden_size, hidden_size)
	l2_2 = nn.LSTM(2 * hidden_size, hidden_size)

	fwdSeq_2 = nn.Sequencer(l2_1)
	bwdSeq_2 = nn.Sequencer(l2_2)
	merge_2 = nn.JoinTable(1, 1)
	mergeSeq_2 = nn.Sequencer(merge_2)

	concat_2 = nn.ConcatTable()
	concat_2:add(fwdSeq_2):add(nn.Sequential():add(nn.ReverseTable()):add(bwdSeq_2):add(nn.ReverseTable()))
	brnn_2 = nn.Sequential()
		:add(concat_2)
		:add(nn.ZipTable())
		:add(mergeSeq_2)

	l3_1 = nn.LSTM(2 * hidden_size, hidden_size)
	l3_2 = nn.LSTM(2 * hidden_size, hidden_size)

	fwdSeq_3 = nn.Sequencer(l3_1)
	bwdSeq_3 = nn.Sequencer(l3_2)
	merge_3 = nn.JoinTable(1, 1)
	mergeSeq_3 = nn.Sequencer(merge_3)

	concat_3 = nn.ConcatTable()
	concat_3:add(fwdSeq_3):add(nn.Sequential():add(nn.ReverseTable()):add(bwdSeq_3):add(nn.ReverseTable()))
	brnn_3 = nn.Sequential()
		:add(concat_3)
		:add(nn.ZipTable())
		:add(mergeSeq_3)

	rnn = nn.Sequential()
		:add(brnn_1)
		:add(brnn_2)
		:add(brnn_3)
		:add(nn.Sequencer(nn.Linear(hidden_size * 2, klass), 1)) -- times two due to JoinTable

	s = use_cuda == true and rnn:cuda() or rnn
end

function model_weibo()
	label_set = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z" }
	klass = table.getn(label_set) + 1
	ksize = 5
	window = 24
	height = 30

	m = nn.Sequential()
	-- first stage: before: 24 * 30; after: 12 * 16
	m:add(nn.SpatialConvolution(3, 16, ksize, ksize, 1, 1, (ksize - 1) / 2, (ksize - 1) / 2 + 1))
	m:add(nn.ReLU())
	m:add(nn.SpatialMaxPooling(2, 2, 2, 2))
	-- second stage: before: 12 * 16; after: 6 * 8
	m:add(nn.SpatialConvolution(16, 32, ksize, ksize, 1, 1, (ksize - 1) / 2, (ksize - 1) / 2))
	m:add(nn.ReLU())
	m:add(nn.SpatialMaxPooling(2, 2, 2, 2))
	-- third stage: before: 6 * 8; after: 3 * 4
	m:add(nn.SpatialConvolution(32, 64, ksize, ksize, 1, 1, (ksize - 1) / 2, (ksize - 1) / 2))
	m:add(nn.ReLU())
	m:add(nn.SpatialMaxPooling(2, 2, 2, 2))
	-- last stage: standard 1-layer mlp
	m:add(nn.Reshape(64 * 3 * 4))
	m:add(nn.Linear(64 * 3 * 4, klass))

	s = use_cuda == true and nn.Sequencer(m):cuda() or nn.Sequencer(m)
end
