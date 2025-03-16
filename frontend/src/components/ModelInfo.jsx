import React from 'react';
import { useNavigate } from 'react-router-dom';

function ModelInfo({ modelInfo, isModelTrained }) {
  const navigate = useNavigate();
  
  if (!isModelTrained) {
    return (
      <div className="model-info-container">
        <div className="card">
          <h2>Model Not Trained</h2>
          <p>Please upload data and train the model first.</p>
          <button 
            onClick={() => navigate('/')} 
            className="submit-button"
          >
            Go to Upload
          </button>
        </div>
      </div>
    );
  }
  
  if (!modelInfo) {
    return (
      <div className="model-info-container">
        <div className="card">
          <h2>Loading Model Information...</h2>
          <div className="loading-spinner-large"></div>
        </div>
      </div>
    );
  }
  
  return (
    <div className="model-info-container">
      <div className="card">
        <h2>LSTM Model Architecture</h2>
        
        <div className="model-architecture">
          <div className="architecture-diagram">
            {modelInfo.architecture.map((layer, index) => (
              <div key={index} className="layer-box">
                <div className={`layer-type ${layer.type.toLowerCase()}`}>
                  {layer.type}
                </div>
                <div className="layer-details">
                  {layer.type === 'LSTM' && (
                    <>
                      <span>Units: {layer.units}</span>
                      <span>Return Sequences: {layer.return_sequences ? 'Yes' : 'No'}</span>
                    </>
                  )}
                  {layer.type === 'Dropout' && (
                    <span>Rate: {layer.rate}</span>
                  )}
                  {layer.type === 'Dense' && (
                    <span>Units: {layer.units}</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
        
        <div className="model-details">
          <h3>Model Configuration</h3>
          <div className="details-grid">
            <div className="detail-item">
              <span className="detail-label">Look Back Window:</span>
              <span className="detail-value">{modelInfo.look_back} time steps</span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Input Features:</span>
              <span className="detail-value">{modelInfo.features.join(', ')}</span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Target Variable:</span>
              <span className="detail-value">Close Price</span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Optimizer:</span>
              <span className="detail-value">Adam</span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Loss Function:</span>
              <span className="detail-value">Mean Squared Error (MSE)</span>
            </div>
          </div>
        </div>
        
        <div className="model-explanation">
          <h3>How LSTM Works for Price Prediction</h3>
          <p>
            Long Short-Term Memory (LSTM) networks are a specialized type of recurrent neural network (RNN) 
            capable of learning long-term dependencies in sequence data. This makes them ideal for time series 
            forecasting like Bitcoin price prediction.
          </p>
          <div className="lstm-features">
            <div className="feature-item">
              <h4>Memory Cells</h4>
              <p>LSTMs contain memory cells that can maintain information for long periods of time, 
              allowing the model to remember patterns from days or weeks ago.</p>
            </div>
            <div className="feature-item">
              <h4>Gates</h4>
              <p>LSTM uses three gates (input, forget, and output) to control the flow of information, 
              deciding what data to keep or discard.</p>
            </div>
            <div className="feature-item">
              <h4>Sequence Learning</h4>
              <p>The model learns from sequences of {modelInfo.look_back} previous time steps to predict 
              the next price, capturing temporal dependencies in the data.</p>
            </div>
          </div>
        </div>
        
        <div className="navigation-buttons">
          <button onClick={() => navigate('/training')} className="nav-button">View Training Results</button>
          <button onClick={() => navigate('/predict')} className="nav-button">Make Predictions</button>
        </div>
      </div>
    </div>
  );
}

export default ModelInfo;