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
	label_set = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z" }
	klass = table.getn(label_set) + 1
	height = 44
	channel = 3
	feature_len = height * channel
	hidden_size = 200

	fwd = nn.LSTM(feature_len, hidden_size)
	fwdSeq = nn.Sequencer(fwd)
	bwd = nn.LSTM(feature_len, hidden_size)
	bwdSeq = nn.Sequencer(bwd)
	merge = nn.JoinTable(1, 1)
	mergeSeq = nn.Sequencer(merge)

	concat = nn.ConcatTable()
	concat:add(fwdSeq):add(nn.Sequential():add(nn.ReverseTable()):add(bwdSeq))
	brnn = nn.Sequential()
		:add(concat)
		:add(nn.ZipTable())
		:add(mergeSeq)

	rnn = nn.Sequential()
		:add(brnn) 
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
