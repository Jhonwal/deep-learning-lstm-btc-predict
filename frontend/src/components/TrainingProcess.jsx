import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { useNavigate } from 'react-router-dom';

function TrainingProcess({ trainResults, isModelTrained }) {
  const navigate = useNavigate();
  
  if (!isModelTrained) {
    return (
      <div className="training-container">
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
  
  if (!trainResults && isModelTrained) {
    return (
      <div className="training-container">
        <div className="card">
          <h2>Training Results Unavailable</h2>
          <p>The model is trained, but detailed training results couldn't be loaded.</p>
          <p>This can happen if the application was restarted and training results weren't saved.</p>
          <p>You can train the model again to see detailed metrics.</p>
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
  
  // Rest of your component remains the same
  // Prepare data for loss chart
  const lossData = trainResults.history.loss.map((loss, index) => ({
    epoch: index + 1,
    train: loss,
    validation: trainResults.history.val_loss[index]
  }));
  
  // Prepare data for prediction charts
  const trainPredictionData = trainResults.visualization.train.dates.map((date, index) => ({
    date,
    actual: trainResults.visualization.train.actual[index],
    predicted: trainResults.visualization.train.predicted[index]
  }));
  
  const testPredictionData = trainResults.visualization.test.dates.map((date, index) => ({
    date,
    actual: trainResults.visualization.test.actual[index],
    predicted: trainResults.visualization.test.predicted[index]
  }));
  
  return (
    <div className="training-container">
      <div className="card">
        <h2>Training Results</h2>
        
        <div className="metrics-container">
          <div className="metric-box">
            <h3>Training RMSE</h3>
            <div className="metric-value">
              ${trainResults.metrics.training_metrics.RMSE.toFixed(2)}
            </div>
          </div>
          <div className="metric-box">
            <h3>Test RMSE</h3>
            <div className="metric-value">
              ${trainResults.metrics.test_metrics.RMSE.toFixed(2)}
            </div>
          </div>
          <div className="metric-box">
            <h3>R² Score</h3>
            <div className="metric-value">
              {trainResults.metrics.test_metrics.R2_Score.toFixed(4)}
            </div>
          </div>
          <div className="metric-box">
            <h3>Directional Accuracy</h3>
            <div className="metric-value">
              {(trainResults.metrics.test_metrics.Directional_Accuracy * 100).toFixed(2)}%
            </div>
          </div>
        </div>
        
        <div className="chart-container">
          <h3>Loss During Training</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart
              data={lossData}
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottomRight', offset: -5 }} />
              <YAxis label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="train" stroke="#8884d8" name="Training Loss" />
              <Line type="monotone" dataKey="validation" stroke="#82ca9d" name="Validation Loss" />
            </LineChart>
          </ResponsiveContainer>
        </div>
        
        <div className="chart-container">
          <h3>Training Set Predictions</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart
              data={trainPredictionData}
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tickFormatter={(value) => value.substring(5)} />
              <YAxis domain={['dataMin - 1000', 'dataMax + 1000']} />
              <Tooltip labelFormatter={(value) => `Date: ${value}`} formatter={(value) => [`$${value.toFixed(2)}`]} />
              <Legend />
              <Line type="monotone" dataKey="actual" stroke="#8884d8" name="Actual Price" />
              <Line type="monotone" dataKey="predicted" stroke="#82ca9d" name="Predicted Price" />
            </LineChart>
          </ResponsiveContainer>
        </div>
        
        <div className="chart-container">
          <h3>Test Set Predictions</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart
              data={testPredictionData}
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tickFormatter={(value) => value.substring(5)} />
              <YAxis domain={['dataMin - 1000', 'dataMax + 1000']} />
              <Tooltip labelFormatter={(value) => `Date: ${value}`} formatter={(value) => [`$${value.toFixed(2)}`]} />
              <Legend />
              <Line type="monotone" dataKey="actual" stroke="#8884d8" name="Actual Price" />
              <Line type="monotone" dataKey="predicted" stroke="#82ca9d" name="Predicted Price" />
            </LineChart>
          </ResponsiveContainer>
        </div>
        
        <div className="navigation-buttons">
          <button onClick={() => navigate('/model')} className="nav-button">View Model Info</button>
          <button onClick={() => navigate('/predict')} className="nav-button">Make Predictions</button>
        </div>
      </div>
    </div>
  );
}

export default TrainingProcess;