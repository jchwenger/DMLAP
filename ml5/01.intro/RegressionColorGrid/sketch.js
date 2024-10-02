/*
  Data and machine learning for creative practice (DMLCP)
  Regression example 2

  In this code we create colour markers on the screen, each time saving their
  coordinates and r,g,b values and providing them to our neural network.
  e.g. nn.addData(inputs, output); in mousePressed();

  Once we have enough points we call nnTrain(); this trains the neural network

  The demo will show a grid with all the colors predicted at each position.

  Instructions:
  - Click to place a point,
  - Press 1 to 3 to change the colour of the points
  - Press 't' to train the model

  All information to customise your neural network and training can be found here:
  https://docs.ml5js.org/#/reference/neural-network

  IDEA:
  - Can you find patterns where the neural net fails to capture the data?
  - Maybe you would like to develop this sketch to allow you to explore the different parameters
    of your network and of training more easily? One way of doing that would be to work with a
    GUI library like https://github.com/bitcraftlab/p5.gui
  - In this vein, one could imagine implementing ways to:
    - Show which class is currently selected (1, 2 or 3) somewhere on the canvas
    - Reset the neural network when pressing a key
    - Reset the datapoints when pressing another key
    - Show the prediction for the pixel under the mouse (draw an ellipse with the predicted color?)
    - If you wanted to save/load your data points, here's how to do it:

      // This will actually download a file.
      function saveData(path){
        saveJSON(data_points, './data.json');
      }

      // Loading and saving
      function loadData(path){
        function loaded(json){
          // Note: you might want to recreate your nn here (or not)
          data_points = json;              // Load existing data if any
          for (const data of data_points){ // if we have points add them
            let inputs = [data.x, data.y];
            let outputs = [data.r, data.g, data.b];
            nn.addData(inputs, outputs); // this adds the data to the neural network
          }
        }
        loadJSON(path, loaded);
      }

*/

let nn,
    ourColor;
    mode = "training";

let data_points = [];

let canvasW = 500,
    canvasH = 500;

function setup() {
  createCanvas(canvasW, canvasH);

  // For this example to work across all browsers
  // "webgl" or "cpu" needs to be set as the backend
  ml5.setBackend("webgl");

  // set our background once at the start
  background(200);

  ourColor = color(255,10,10);

  // Setup the neural network
  // For each example, the network has two inputs [x, y] (the mouse position)
  // and three outputs [r, g, b] (the corresponding color)
  // Here we use the default config. To add more, look here under 'regression':
  // https://archive-docs.ml5js.org/#/reference/neural-network?id=defining-custom-layers
  // The new doc hasn't got as much information:
  // https://docs.ml5js.org/#/reference/neural-network?id=ml5neuralnetwork
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
  // Note in this demo we update the canvas every frame
  background(200);

  fill(0);
  noStroke();
  text("Click to place a point\nPress 1-3 to change color\nPress 't' to train.", 20, 20);

  fill(ourColor);
  stroke(1);
  ellipse(mouseX, mouseY, 5, 5);

  // Draw the grid
   if (mode === "demo") {
     drawGrid();
   }

  // Draw our data points
  for (const p of data_points){
    fill(p.r, p.g, p.b);
    stroke(0);
    ellipse(p.x, p.y, 10, 10);
  }
}

function mousePressed() {
  // IDEA: perhaps we want to modify the sketch so that it is possible to
  // add more data after training, and train again?
  if (mode == "training") {
    // Draw a circle at our mouse coordinates, set to the colour of our current color.
    // let inputs = [mouseX/canvasW, mouseY/canvasH];
    let inputs = [mouseX, mouseY];
    let outputs = [red(ourColor), green(ourColor), blue(ourColor)];
    data_points.push({x: mouseX, y: mouseY,
                      r: red(ourColor), g: green(ourColor), b: blue(ourColor)});
    nn.addData(inputs, outputs);
    // console.log(ourColor, inputs, outputs);
  }
}

function keyPressed() {
  // Switch is equivalent to a series of if-else statements
  switch (key) {
    // IDEA: add more colours?
    // NOTE: as I'm writing this, training will fail if one of the three inputs
    //       has the same value in every data point (e.g. 10 data points, and
    //       only ever '10' for the R value in each). Therefore, I use '1', '2',
    //       and '3' as arbitrary different values.
    case "1":
      ourColor = color(255,1,1);
      break; // Important to call break after each "case" otherwise execution will continue to the next
    case "2":
      ourColor = color(2,255,2);
      break;
    case "3":
      ourColor = color(3,3,255);
      break;
    // "t" starts training the neural network! (see brain.js)
    case "t":
      // WHERE THE TRAINING HAPPENS: no need to change this
      // More information here: https://docs.ml5js.org/#/reference/neural-network?id=train-the-model
      nn.normalizeData();    // Normalise our data
      nn.train({
        epochs: 35,          // This controls for how long we train!
        batchSize: 12,       // We will see what all this means soon
        learningRate: 0.2,
        validationSplit: 0.0 // By default we won't perform any validation (this does not influence training, but removes datapoints if > 0)
      }, finishedTraining);  // This is a callback: the function finishedTraining is called when the training is over
      mode = "demo";         // This signals that we want to see updates as we train (might slow down training)
      break;
  }
}

// Draw the grid
function drawGrid() {
  let step = 20; // The square size for the grid

  // Collect center points of each cell and put the points in an array
  // (the array will be ordered by rows of the grid)
  let inputs = [];
  for (let y = 0; y < canvasW; y += step){
    for (let x = 0; x < canvasH; x += step){
      inputs.push([(x + step/2), (y + step/2)]);
    }
  }

  // Predict colors for ALL collected center points
  let res = nnPredict(inputs);

  // Now draw the grid with the predicted colors
  let i = 0;
  for (let y = 0; y < canvasW; y += step){
    for (let x = 0; x < canvasH; x += step){
      const clr = res[i];
      fill(clr[0], clr[1], clr[2]);
      rect(x, y, step, step);
      i += 1; // Increment cell (recall these are ordered by row and the inner loop
    }         // goes along a row)
  }
}

// NN things
function finishedTraining(){
  console.log("We finished training!");
  console.log("Here's how one raw prediction looks like for one datapoint:");
  console.log(nn.predictSync([width/2,height/2])); // prediction for the centre of the canvas
}

function nnPredict(input) {

  // The standard ML5js way to perform predictions is asyncronous, but async
  // programming can be tricky. Luckily the "predictSync" function forces ML5js
  // to give us the result immediately and simplifies matters
  const res = nn.predictSync(input);

  // ML5 places each component of the prediction into an object.
  // As a result, the format of the output of a prediction is different from
  // the one given as training data. E.g. if we trained the network with one
  // input array [x] and an output array [a, b, c] one prediction will give
  // [{0:predicted_a, label: 0, value:predicted_a, unNormalizedValue: unnormalized_predicted_a},
  //  {1:predicted_b, label: 1, value:predicted_b, unNormalizedValue: unnormalized_predicted_b},
  //  {2:predicted_c, label: 2, value:predicted_c, unNormalizedValue: unnormalized_predicted_c}]
  // meaning that with "straight" ML5js we will need to parse these values and
  // put them into our preferred format (which is likely the one we provided in
  // the first place as training data).

  // If we provide an array of inputs (asking for multiple predictions),
  // predictedSync will return an array of such objects, so here we handle the
  // two cases making it consistent with the input format
  if (Array.isArray(input[0])){
    // multiple predictions
    return res.map(element => element.map(v => v.value));
  } else {
    // make one prediction
    return res.map(v => v.value);
  }
}
