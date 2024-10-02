// Import p5.js as an ES module
import p5 from 'https://cdn.skypack.dev/p5';

// Import FilesetResolver and LlmInference from MediaPipe Tasks GenAI
import { FilesetResolver, LlmInference } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai';

new p5((sketch) => {
  let inputTextArea;
  let submitButton;
  let generatedText = '';

  const modelFileName = 'gemma-2b-it-gpu-int4.bin'; // Update the file name as needed
  let llmInference;

  /**
   * Display newly generated partial results in the output text area.
   */

  function displayPartialResults(partialResults, complete) {

    // Append the new partial results to the generatedText
    generatedText += partialResults;

    if (complete) {
      if (!generatedText) {
        generatedText = '(No answer...)';
      }
      submitButton.disabled = false;
    }
  }

  /**
   * Setup function for p5.js sketch.
   */
  sketch.setup = () => {
    // sketch.noCanvas(); // No need for a canvas in this demo
    const c = sketch.createCanvas(600, 300); // Adjust width and height as needed

    // Example adjustments
    sketch.createCanvas(800, 400); // Larger canvas
    sketch.background(175,238,238);

    sketch.fill(0); // Set text color to black
    sketch.textSize(16); // Set the text size
    sketch.textAlign(sketch.LEFT, sketch.TOP); // Align text to the top-left corner

    // HTML business ------------------------------------------------------------------

    // Retrieve the parent of the canvas (main)
    const main = c.parent();

    // Create Input section
    const inputControls = document.createElement('div');
    inputControls.id = "input-controls";
    inputTextArea = document.createElement('textarea');
    inputTextArea.id = "input";
    inputTextArea.placeholder = "Input text here and the response will appear in the sketch...";

    // Create Submit button
    submitButton = document.createElement('button');
    submitButton.id = "input-button";

    // submitButton
    submitButton.disabled = true; // Disable until model loads

    submitButton.onclick = () => {
      generatedText = ''; // Clear previous results
      submitButton.disabled = true;
      llmInference.generateResponse(inputTextArea.value, displayPartialResults);
    };

    submitButton.innerText = 'Loading the model...';

    inputControls.appendChild(inputTextArea);
    inputControls.appendChild(submitButton);

    // Append the container into main
    main.appendChild(inputControls)

    // --------------------------------------------------------------------------------

    // Load the LLM model
    FilesetResolver.forGenAiTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai/wasm')
      .then(genaiFileset => {
        return LlmInference.createFromOptions(genaiFileset, {
          baseOptions: { modelAssetPath: modelFileName },
          // Add additional options here if needed
        });
      })
      .then(llm => {
        llmInference = llm;
        // Now we are ready to chat, activate the button
        submitButton.disabled = false;
        submitButton.innerText = 'Send';
      })
      .catch((e) => {
        console.error(e);
        alert('Failed to load the model.');
      });
  };

  sketch.draw = () => {

    sketch.background(175,238,238);

    // Use text wrapping by specifying a width and height for the text box
    sketch.text(generatedText, 10, 10, sketch.width - 20, sketch.height - 20);
  };

});
