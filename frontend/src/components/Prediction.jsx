import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

function Prediction({ isModelTrained }) {
  const navigate = useNavigate();
  const fileInputRef = useRef(null);
  const [inputData, setInputData] = useState([
    { open: '', high: '', low: '', close: '', volume: '' }
  ]);
  const [lastDate, setLastDate] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showHelp, setShowHelp] = useState(false);
  
  if (!isModelTrained) {
    return (
      <div className="prediction-container">
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
  
  const handleInputChange = (index, field, value) => {
    const newInputData = [...inputData];
    newInputData[index][field] = value;
    setInputData(newInputData);
  };
  
  const addRow = () => {
    setInputData([...inputData, { open: '', high: '', low: '', close: '', volume: '' }]);
  };
  
  const removeRow = (index) => {
    if (inputData.length > 1) {
      const newInputData = [...inputData];
      newInputData.splice(index, 1);
      setInputData(newInputData);
    }
  };
  
  const handleCsvUpload = async (e) => {
    e.preventDefault();
    const file = fileInputRef.current.files[0];
    
    if (!file) {
      setError('Please select a CSV file');
      return;
    }
    
    if (!file.name.endsWith('.csv')) {
      setError('Please upload a CSV file');
      return;
    }
    
    setLoading(true);
    setError('');
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const response = await axios.post('http://localhost:5000/api/upload_prediction_data', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      if (response.data.data) {
        // Convert the data to the input format
        const parsedData = response.data.data.map(row => ({
          open: row[0].toString(),
          high: row[1].toString(),
          low: row[2].toString(),
          close: row[3].toString(),
          volume: row[4].toString()
        }));
        
        setInputData(parsedData);
        
        if (response.data.last_date) {
          setLastDate(response.data.last_date);
        }
        
        setError('');
      }
    } catch (error) {
      console.error('Error uploading CSV:', error);
      setError(error.response?.data?.error || 'Failed to process CSV file');
    } finally {
      setLoading(false);
    }
  };
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Validate inputs
    const isValid = inputData.every(row => 
      row.open !== '' && row.high !== '' && row.low !== '' && 
      row.close !== '' && row.volume !== ''
    );
    
    if (!isValid) {
      setError('Please fill all fields or upload a valid CSV file');
      return;
    }
    
    if (!lastDate) {
      setError('Please enter the last date');
      return;
    }
    
    setLoading(true);
    setError('');
    
    try {
      // Format data for API
      const formattedData = inputData.map(row => [
        parseFloat(row.open),
        parseFloat(row.high),
        parseFloat(row.low),
        parseFloat(row.close),
        parseFloat(row.volume)
      ]);
      
      const response = await axios.post('http://localhost:5000/api/predict', {
        data: formattedData,
        last_date: lastDate
      });
      
      setPrediction(response.data);
    } catch (error) {
      console.error('Error making prediction:', error);
      setError(error.response?.data?.error || 'Failed to make prediction');
    } finally {
      setLoading(false);
    }
  };
  
  const handleClear = () => {
    setInputData([{ open: '', high: '', low: '', close: '', volume: '' }]);
    setLastDate('');
    setPrediction(null);
    setError('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };
  
  // Sample data for demonstration
  const handleFillSample = () => {
    setInputData([
      { open: '16500.25', high: '16750.45', low: '16450.30', close: '16700.10', volume: '1250.75' },
      { open: '16700.10', high: '16900.20', low: '16600.50', close: '16850.75', volume: '1320.40' },
      { open: '16850.75', high: '17100.30', low: '16800.25', close: '17050.60', volume: '1450.25' },
    ]);
    
    // Set yesterday's date
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);
    setLastDate(yesterday.toISOString().split('T')[0]);
  };
  
  // Prepare chart data
  const prepareChartData = () => {
    if (!prediction) return [];
    
    // Get historical data
    const historicalData = inputData.map((row, index) => {
      // Calculate date based on last date and index
      const date = new Date(lastDate);
      date.setDate(date.getDate() - (inputData.length - index));
      
      return {
        date: date.toISOString().split('T')[0],
        price: parseFloat(row.close)
      };
    });
    
    // Add prediction
    return [
      ...historicalData,
      {
        date: prediction.date,
        price: null,
        predictedPrice: prediction.predicted_price
      }
    ];
  };
  
  const chartData = prepareChartData();
  
  return (
    <div className="prediction-container">
      <div className="card">
        <h2>Make Bitcoin Price Predictions</h2>
        <p>
          Upload a CSV file with Bitcoin price data or enter data manually to predict the next day's closing price.
          <button className="help-button" onClick={() => setShowHelp(!showHelp)}>
            {showHelp ? 'Hide Help' : 'Show Help'}
          </button>
        </p>
        
        {showHelp && (
          <div className="help-container">
            <h3>How to Use</h3>
            <ol>
              <li>Upload a CSV file with Bitcoin price data (format: date,open,high,low,close,volume) OR</li>
              <li>Enter at least the last 3 days of Bitcoin pricing data manually (more is better)</li>
              <li>Ensure the last date field is set correctly</li>
              <li>Click "Predict" to generate a price forecast for the next day</li>
            </ol>
            <p>You can click "Sample Data" to fill the form with example data for testing.</p>
          </div>
        )}
        
        <div className="csv-upload-section">
          <h3>Upload CSV Data</h3>
          <p>CSV format should have columns: date,open,high,low,close,volume</p>
          <form onSubmit={handleCsvUpload} className="upload-form">
            <div className="file-upload-container">
              <input
                type="file"
                id='file-upload'
                accept=".csv"
                ref={fileInputRef}
                className="file-csv-waguer"
              />
              <button 
                type="submit" 
                className="upload-button"
                disabled={loading}
              >
                {loading ? <span className="loading-spinner"></span> : 'Process CSV'}
              </button>
            </div>
          </form>
        </div>
        
        <form onSubmit={handleSubmit}>
          <div className="date-input">
            <label>Last Date:</label>
            <input
              type="date"
              value={lastDate}
              onChange={(e) => setLastDate(e.target.value)}
              required
            />
          </div>
          
          <div className="manual-input-section">
            <h3>Or Enter Data Manually</h3>
            <div className="data-input-container">
              <table className="data-input-table">
                <thead>
                  <tr>
                    <th>Open ($)</th>
                    <th>High ($)</th>
                    <th>Low ($)</th>
                    <th>Close ($)</th>
                    <th>Volume</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {inputData.map((row, index) => (
                    <tr key={index}>
                      <td>
                        <input
                          type="number"
                          min="0"
                          step="0.01"
                          value={row.open}
                          onChange={(e) => handleInputChange(index, 'open', e.target.value)}
                          placeholder="Open"
                        />
                      </td>
                      <td>
                        <input
                          type="number"
                          min="0"
                          step="0.01"
                          value={row.high}
                          onChange={(e) => handleInputChange(index, 'high', e.target.value)}
                          placeholder="High"
                        />
                      </td>
                      <td>
                        <input
                          type="number"
                          min="0"
                          step="0.01"
                          value={row.low}
                          onChange={(e) => handleInputChange(index, 'low', e.target.value)}
                          placeholder="Low"
                        />
                      </td>
                      <td>
                        <input
                          type="number"
                          min="0"
                          step="0.01"
                          value={row.close}
                          onChange={(e) => handleInputChange(index, 'close', e.target.value)}
                          placeholder="Close"
                        />
                      </td>
                      <td>
                        <input
                          type="number"
                          min="0"
                          step="0.01"
                          value={row.volume}
                          onChange={(e) => handleInputChange(index, 'volume', e.target.value)}
                          placeholder="Volume"
                        />
                      </td>
                      <td>
                        <button
                          type="button"
                          onClick={() => removeRow(index)}
                          className="remove-button"
                          disabled={inputData.length <= 1}
                        >
                          ✕
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
            <div className="button-group">
              <button type="button" onClick={addRow} className="add-button">
                + Add Row
              </button>
              <button type="button" onClick={handleFillSample} className="sample-button">
                Fill Sample Data
              </button>
            </div>
          </div>
          
          {error && <div className="error-message">{error}</div>}
          
          <div className="button-group">
            <button type="submit" className="submit-button" disabled={loading}>
              {loading ? <span className="loading-spinner"></span> : 'Predict'}
            </button>
            <button type="button" onClick={handleClear} className="clear-button">
              Clear
            </button>
          </div>
        </form>
        
        {prediction && (
          <div className="prediction-result">
            <h3>Prediction Result</h3>
            <div className="prediction-value">
              <span className="prediction-label">Predicted Price for {prediction.date}:</span>
              <span className="prediction-price">${prediction.predicted_price.toFixed(2)}</span>
            </div>
            
            <div className="chart-container">
              <h4>Price Trend with Prediction</h4>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart
                  data={chartData}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis domain={['dataMin - 500', 'dataMax + 500']} />
                  <Tooltip 
                    formatter={(value) => value ? [`${value.toFixed(2)}`] : ['']}
                    labelFormatter={(value) => `Date: ${value}`}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="price" stroke="#8884d8" name="Historical Price" connectNulls />
                  <Line type="monotone" dataKey="predictedPrice" stroke="#82ca9d" name="Predicted Price" strokeDasharray="5 5" connectNulls />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default Prediction;