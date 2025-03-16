import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import axios from 'axios';
import './App.css';

import Header from './components/Header';
import UploadData from './components/UploadData';
import TrainingProcess from './components/TrainingProcess';
import ModelInfo from './components/ModelInfo';
import Prediction from './components/Prediction';

function App() {
  const [trainResults, setTrainResults] = useState(null);
  const [isModelTrained, setIsModelTrained] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  
  useEffect(() => {
    // Check if model is trained
    const checkModelStatus = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/model-info');
        setIsModelTrained(response.data.trained);
        if (response.data.trained) {
          setModelInfo(response.data);
          
          // If model is trained, also fetch training results
          try {
            const resultsResponse = await axios.get('http://localhost:5000/api/training-results');
            setTrainResults(resultsResponse.data);
          } catch (error) {
            console.error('Error fetching training results:', error);
            // We don't set trainResults to anything here, it remains null
          }
        }
      } catch (error) {
        console.error('Error checking model status:', error);
      }
    };
    
    checkModelStatus();
  }, []);
  
  return (
    <Router>
      <div className="app">
        <Header />
        <div className="nav-container">
          <nav className="main-nav">
            <Link to="/" className="nav-link">Upload Data</Link>
            <Link to="/training" className="nav-link">Training</Link>
            <Link to="/model" className="nav-link">Model Info</Link>
            <Link to="/predict" className="nav-link">Predict</Link>
          </nav>
        </div>
        
        <main className="main-content">
          <Routes>
            <Route path="/" element={<UploadData 
              setTrainResults={setTrainResults} 
              setIsModelTrained={setIsModelTrained} 
              setModelInfo={setModelInfo} 
            />} />
            <Route path="/training" element={<TrainingProcess 
              trainResults={trainResults} 
              isModelTrained={isModelTrained} 
            />} />
            <Route path="/model" element={<ModelInfo 
              modelInfo={modelInfo} 
              isModelTrained={isModelTrained} 
            />} />
            <Route path="/predict" element={<Prediction 
              isModelTrained={isModelTrained} 
            />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;