import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

function UploadData({ setTrainResults, setIsModelTrained, setModelInfo }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();
  
  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setError('');
  };
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!file) {
      setError('Please select a file');
      return;
    }
    
    if (!file.name.endsWith('.csv')) {
      setError('Only CSV files are supported');
      return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    setLoading(true);
    
    try {
      const response = await axios.post('http://localhost:5000/api/train', formData);
      setTrainResults(response.data);
      setIsModelTrained(true);
      
      // Get updated model info
      const modelInfoResponse = await axios.get('http://localhost:5000/api/model-info');
      setModelInfo(modelInfoResponse.data);
      
      navigate('/training');
    } catch (error) {
      console.error('Error training model:', error);
      setError(error.response?.data?.error || 'Failed to train model');
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="upload-container">
      <div className="card">
        <h2>Upload Bitcoin Price Data</h2>
        <p>Upload a CSV file containing Bitcoin price data with columns: 'Open time', 'Open', 'High', 'Low', 'Close', 'Volume'</p>
        
        <form onSubmit={handleSubmit}>
          <div className="file-input-container">
            <label htmlFor="file-upload" className="file-input-label">
              {file ? file.name : 'Choose File'}
            </label>
            <input
              id="file-upload"
              type="file"
              onChange={handleFileChange}
              accept=".csv"
              className="file-input"
            />
          </div>
          
          {error && <div className="error-message">{error}</div>}
          
          <button type="submit" className="submit-button" disabled={loading}>
            {loading ? (
              <span className="loading-spinner"></span>
            ) : (
              'Train Model'
            )}
          </button>
        </form>
        
        <div className="data-info">
          <h3>Expected Data Format:</h3>
          <table className="data-table">
            <thead>
              <tr>
                <th>Open time</th>
                <th>Open</th>
                <th>High</th>
                <th>Low</th>
                <th>Close</th>
                <th>Volume</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>2023-01-01</td>
                <td>16500.25</td>
                <td>16750.45</td>
                <td>16450.30</td>
                <td>16700.10</td>
                <td>1250.75</td>
              </tr>
              <tr>
                <td>...</td>
                <td>...</td>
                <td>...</td>
                <td>...</td>
                <td>...</td>
                <td>...</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default UploadData;