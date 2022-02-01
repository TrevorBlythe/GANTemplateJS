const convnetjs = require('./convnetjs');
const fs = require('fs');
const Jimp = require('jimp');

var out = function (text) {
	console.log(text + ' ');
};

var clamp = function (num) {
	return Math.min(Math.max(num, 0), 1);
};

var clear = function () {
	console.clear();
};

let newImage = new Jimp(50, 50, 'white', (err, image) => {
	if (err) throw err;
});

var startFromScratch = false;
var generator = new convnetjs.Net();
var disc = new convnetjs.Net();

if (!startFromScratch) {
	fs.readFile('models/gen200.json', function (err, data) {
		var json = JSON.parse(data.toString()); // creates json object out of a string
		generator.fromJSON(json);
	});

	fs.readFile('models/disc200.json', function (err, data) {
		var json = JSON.parse(data.toString()); // creates json object out of a string
		disc.fromJSON(json);
	});
} else {
	var layer_defs = [];
	layer_defs.push({ type: 'input', out_sx: 1, out_sy: 1, out_depth: 5 });
	layer_defs.push({ type: 'fc', num_neurons: 70, activation: 'tanh' });
	layer_defs.push({ type: 'fc', num_neurons: 70, activation: 'tanh' });
	// layer_defs.push({ type: 'fc', num_neurons: 10 * 10, activation: 'sigmoid' });
	// layer_defs.push({ type: 'fc', num_neurons: 10 * 10, activation: 'sigmoid' });
	layer_defs.push({ type: 'regression', num_neurons: 50 * 50 * 3 });

	var layer_defsTwo = [];
	layer_defsTwo.push({
		type: 'input',
		out_sx: 50,
		out_sy: 50,
		out_depth: 3,
	});
	// layer_defsTwo.push({
	// 	type: 'conv',
	// 	sx: 5,
	// 	filters: 5,
	// 	stride: 1,
	// 	pad: 2,
	// 	activation: 'relu',
	// });
	// layer_defsTwo.push({ type: 'pool', sx: 2, stride: 2 });
	layer_defsTwo.push({
		type: 'fc',
		num_neurons: 50,
		activation: 'tanh',
	});
	layer_defsTwo.push({
		type: 'fc',
		num_neurons: 50,
		activation: 'tanh',
	});
	// layer_defsTwo.push({ type: 'fc', num_neurons: 50, activation: 'sigmoid' });
	// layer_defsTwo.push({ type: 'fc', num_neurons: 20, activation: 'sigmoid' });
	layer_defsTwo.push({ type: 'regression', num_neurons: 1 });

	generator.makeLayers(layer_defs);
	disc.makeLayers(layer_defsTwo);
}
trainer = new convnetjs.SGDTrainer(generator, {
	method: 'adadelta',
	batch_size: 2,
	learning_rate: 100,

	l2_decay: 0.001,
});

trainerTwo = new convnetjs.SGDTrainer(disc, {
	method: 'adadelta',
	batch_size: 10,
	learning_rate: 100,
	l2_decay: 0.001,
});

var saveCounter = 0;
var epoch = 0;

let loop = function () {
	Jimp.read(
		'images/' + Math.floor(Math.random() * (501 - 1) + 1) + '.png',

		//'images/' + 4 + '.png',
		(err, imagey) => {
			if (err) throw err;
			let pixArr = [];
			for (var i = 0; i < 50; i++) {
				for (var j = 0; j < 50; j++) {
					let temp = Jimp.intToRGBA(imagey.getPixelColor(i, j));
					//temp is {r:0, b:0, g:0, a:0} but with whatevs nums
					pixArr.push(temp.r / 255);
					pixArr.push(temp.g / 255);
					pixArr.push(temp.b / 255);
				}
			}

			Jimp.read(
				'images/' + Math.floor(Math.random() * (501 - 1) + 1) + '.png',
				//'images/' + 4 + '.png',

				(err, imagey) => {
					if (err) throw err;
					let pixArrTwo = [];
					for (var i = 0; i < 50; i++) {
						for (var j = 0; j < 50; j++) {
							let temp = Jimp.intToRGBA(imagey.getPixelColor(i, j));
							//temp is {r:0, b:0, g:0, a:0} but with whatevs nums

							pixArrTwo.push(temp.r / 255);
							pixArrTwo.push(temp.g / 255);
							pixArrTwo.push(temp.b / 255);
						}
					}

					pixArr = new convnetjs.Vol(pixArr);

					pixArrTwo = new convnetjs.Vol(pixArrTwo);

					let vector = new convnetjs.Vol(1, 1, 5);
					let image = generator.forward(vector);
					trainerTwo.train(new convnetjs.Vol([...image.w]), [0]);

					vector = new convnetjs.Vol(1, 1, 5);
					image = generator.forward(vector);
					trainerTwo.train(new convnetjs.Vol([...image.w]), [0]);

					vector = new convnetjs.Vol(1, 1, 5);
					image = generator.forward(vector);
					trainerTwo.train(new convnetjs.Vol([...image.w]), [1]);
					let expected = new Float64Array(50 * 50 * 3);
					for (var i = 0; i < expected.length; i++) {
						expected[i] = image.w[i] - disc.layers[0].in_act.dw[i];
					}

					trainer.train(vector, expected);
					trainerTwo.train(pixArr, [1]);
					trainerTwo.train(pixArrTwo, [1]);

					saveCounter++;
					epoch++;

					if (epoch % 100 == 0) {
						console.log('making image');
						var x = new convnetjs.Vol(1, 1, 5);
						var imagey = generator.forward(x).w;
						// console.log(imagey);

						for (var i = 0; i < 50; i++) {
							for (var j = 0; j < 50; j++) {
								let r = clamp(imagey[3 * (i * 50 + j)]);
								let g = clamp(imagey[3 * (i * 50 + j) + 1]);
								let b = clamp(imagey[3 * (i * 50 + j) + 2]);

								color = Jimp.rgbaToInt(r * 255, g * 255, b * 255, 255);
								newImage.setPixelColor(color, i, j);
							}
						}
						newImage.write('imagey.png');
					}

					if (saveCounter % 100 == 0) {
						fs.writeFileSync(
							'models/gen' + saveCounter + '.json',
							JSON.stringify(generator.toJSON()),
							(err) => {
								if (err) {
									console.error('model save error: ' + err);
									return;
								}
							}
						);
						fs.writeFileSync(
							'models/disc' + saveCounter + '.json',
							JSON.stringify(disc.toJSON()),
							(err) => {
								if (err) {
									console.error('model save error: ' + err);
									return;
								}
							}
						);
						if (saveCounter > 1000) {
							saveCounter = 0;
						}
						setImmediate(loop);
					} else {
						setImmediate(loop);
					}
				}
			);
		}
	);
};

setInterval(loop, 100);
