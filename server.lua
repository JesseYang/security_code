require 'main'
local app = require('waffle')

app.post('/upload', function(req, res)
	local img = req.form.file:toImage()
	img = (img * 255):byte()
	img = img:transpose(1, 2):transpose(2, 3)
	local ret = recognize(img)
	res.send(ret)
end)

app.get('/upload', function(req, res)
	res.render('upload.html', { })
end)

app.listen({host="10.50.101.163", port="8080"})

