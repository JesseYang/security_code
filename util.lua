function mysplit(inputstr, sep)
	if sep == nil then
		sep = "%s"
	end
	local t={ }; i=1
	for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
		t[i] = str
		i = i + 1
	end
	return t
end

function shuffle(array)
	local n, random, j = table.getn(array), math.random
	for i=1, n do
		j,k = random(n), random(n)
		array[j],array[k] = array[k],array[j]
	end
	return array
end