/*
  Data and machine learning for creative practice (DMLCP)
  Regression example 1

  In this code we create colour markers on the screen, each time saving their
  coordinates and r,g,b values and providing them to our neural network.
  e.g. nn.addData(inputs, output); in mousePressed();

  Once we have enough points we call nn.train(); this trains the neural network

  After it has trained we ask the neural network to predict the colour for the
  mouse coordinates (reflected in the square)

  Instructions:
  - Click to place a point,
  - Press 1 to 3 to change the colour of the points
  - Press 't' to train the model

  IDEA: An experiment that could help you gain some practice with this NN system
        could be to try and implement two variants of this sketch: one that takes
        *one* input instead of two (for instance only the x or the y). You would
        need to make changes when you define your network, and whenever an input
        is fed into your nn. You could also try and imagine what system could take
        *three* inputs instead of two?
  IDEA: Similarly, nothing stops you from making the nn predict only one number,
        instead of three. Instead of having three different colours as we have here,
        you could define three different radius sizes for ellipses. You select which
        one using the numbers as above, and when you click to create data, you create
        an ellipse with that radius at that point. Then, you train your neural net to
        predict just this number, the radius, and when the model is trained, you could
        draw an ellipse with the predicted radius at that spot.
*/

let nn,
    ourColor,
    mode = "training";

function setup() {
  createCanvas(500, 500);

  // For this example to work across all browsers
  // "webgl" or "cpu" needs to be set as the backend
  ml5.setBackend("webgl");

  // set our background once at the start
  background(0);

  // our first colour can be red;
  ourColor = color(255,1,1);

  // note in this demo we just clear the canvas at the beginning
  fill(255);
  noStroke();
  text("Click to place a point\nPress 1-3 to change color\nPress 't' to train.", 20, 20);

  // Setup the neural network
  // For each example, the network has two inputs [x, y] (the mouse position)
  // and three outputs [r, g, b] (the corresponding color)
  // (Here we use the default config. To add more, look here under 'regression':
  // https://archive-docs.ml5js.org/#/reference/neural-network?id=defining-custom-layers)
  nn = ml5.neuralNetwork({
    inputs: 2,          // two inputs: x and y
    outputs: 3,         // three outputs: r, g and b
    task: 'regression', // because we predict the three numbers directly (r/g/b values)
    debug: true,        // this opens the training pane
    // this is the default settings, see: https://github.com/ml5js/ml5-next-gen/blob/9aff062c6b2f999d7d5e98cc03619eb3695e3826/src/NeuralNetwork/index.js#L700
    // (you can add more 'dense' layers with more/less units!)
    // (all choices of activations can be found here: https://github.com/tensorflow/tfjs/blob/master/tfjs-layers/src/activations.ts)
    // layers: [           
    //   { type: 'dense', units: 16, activation: 'relu', }, // try activation: 'tanh'
    //   { type: 'dense', activation: 'sigmoid', } // this outputs numbers between 0 and 1
    // ]
  });

}

function draw() {
  // if we have already trained we want to show the output as a square
  // in the bottom right-hand side
   if (mode === "demo") {
     const mouseColor = nnPredict([mouseX, mouseY]);

     // IDEA: why not use the colour for the background, or a frame around the canvas?
     // IDEA: how about changing this piece of code so that an ellipse is
     //       drawn where the mouse is, with the color displayed accordingly?
     // IDEA: you could also imagine looking into coding a bouncing ball (or
     //       several), and use the position to determine the colour...
     //       (of course, in both these cases you might want to hide the data points)
     // console.log('mouse color:', mouseColor);
     fill(mouseColor[0], mouseColor[1], mouseColor[2]);
     rect(0, height-50, 50, 50);
   }
}

function mousePressed() {
  if (mode == "training") {
    // draw a circle at our mouse coordinates, set to the colour of our current color.
    fill(ourColor);
    noStroke();
    ellipse(mouseX, mouseY, 10, 10);

    let inputs = [mouseX, mouseY];
    let outputs = [
      red(ourColor),
      green(ourColor),
      blue(ourColor)
    ];

    // this adds the data to the neural network
    nn.addData(inputs, outputs);
    // console.log(ourColor, inputs, outputs);
  }
}

function keyPressed() {
  // IDEA: play with more colours?
  // NOTE: as I'm writing this, training will fail if one of the three inputs
  //       has the same value in every data point (e.g. 10 data points, and
  //       only ever '10' for the R value in each). Therefore, I use '1', '2',
  //       and '3' as arbitrary different values.
  if (key == "1") {
    ourColor = color(255,1,1); // red
  } else if (key == "2") {
    ourColor = color(2,255,2); // green
  } else if (key == "3") {
    ourColor = color(3,3,255); // blue
  } else if (key == "t") {
    // WHERE THE TRAINING HAPPENS: no need to change this
    // More information here: https://learn.ml5js.org/#/reference/neural-network?id=train
    nn.normalizeData();    // Normalise our data
    nn.train({
      epochs: 35,          // This controls for how long we train!
      batchSize: 12,       // We will see what all this means soon
      learningRate: 0.2,
      validationSplit: 0.0 // By default we won't perform any validation (this does not influence training, but removes datapoints if > 0)
    }, finishedTraining);  // This is a callback: the function finishedTraining is called when the training is over
  }
}

function finishedTraining(){
  console.log("We finished training!");
  console.log("Here's how one raw prediction looks like for one datapoint:");
  console.log(nn.predictSync([width/2,height/2])); // prediction for the centre of the canvas
  // Switch to the demo mode - this will tell our app that we can visualize the result
  mode = "demo";
}

function nnPredict(input) {
  // The standard ML5js way to perform predictions is asyncronous, but async programming can be tricky.
  // Luckily the "predictSync" function forces ML5js to give us the result immediately and simplifies matters
  const res = nn.predictSync(input);

  // ML5 places each component of the prediction into an object.
  // as a result, the format of the output of a prediction is different from the one given as training data
  // E.g. if we trained the network with one input array [x] and an output array [a, b, c] one prediction will give
  // [{0:predicted_a, label: 0, value:predicted_a, unNormalizedValue: unnormalized_predicted_a},
  //  {1:predicted_b, label: 1, value:predicted_b, unNormalizedValue: unnormalized_predicted_b},
  //  {2:predicted_c, label: 2, value:predicted_c, unNormalizedValue: unnormalized_predicted_c}]
  // meaning that with "straight" ML5js we will need to parse these values and put them into our preferred format
  // (which is likely the one we provided in the first place as training data).
  //
  // If we provide an array of inputs (asking for multiple predictions), predictedSync will return an array of such objects, so here we handle the two cases

  // making it consistent with the input format
  if (Array.isArray(input[0])){
    // multiple predictions
    return res.map(element => element.map(v => v.value));
  } else {
    // make one prediction
    return res.map(v => v.value);
  }
}
