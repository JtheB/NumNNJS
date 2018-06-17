const numjs = require('numjs');
const lychee = require('lycheejs')(__dirname);

lychee.environment.init(function(sandbox) {
  let _Mersenne = lychee.import('lychee.math.Mersenne');
  let twister = new _Mersenne({
    seed: 42
  });

  var inputArray = numjs.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
  ]);

  var outputArray = numjs.array([
    [0, 0, 1, 1]
  ]).T;
  console.log("What it should be: ");
  console.log(outputArray);

  let weights = numjs.array([
    [twister.random()],
    [twister.random()],
    [twister.random()]
  ]);

  function sigmoidDerivative(array) {
    let ones = numjs.ones(array.shape);
    return array.multiply(ones.subtract(array));
  }

  function train(iterationsCounter) {
    for (let i = 0; i < iterationsCounter; i++) {
      let hiddenLayer = numjs.sigmoid(numjs.dot(inputArray, weights));
      let hiddenLayerError = outputArray.subtract(hiddenLayer);
      let hiddenLayerDelta = hiddenLayerError.multiply(sigmoidDerivative(hiddenLayer));
      weights.add(numjs.dot(inputArray.T, hiddenLayerDelta));
    }
  }

  train(10000);
  console.log("After Training: ");
  let output = numjs.sigmoid(numjs.dot(inputArray, weights));
  console.log(output);
});
