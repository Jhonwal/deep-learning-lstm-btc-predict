import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, ComposedChart } from 'recharts';
import { Clock, TrendingUp, BarChart2, Award, RefreshCw, Info, ChevronDown, ChevronUp } from 'lucide-react';

export default function App() {
  const [modelInfo, setModelInfo] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showDetails, setShowDetails] = useState({
    model: false,
    metrics: false,
    evaluation: false
  });
  
  // Format for currency display
  const formatter = new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2
  });
  
  // Calculate price change percentage
  const calculatePriceChange = () => {
    if (!predictions || predictions.predictions.length < 2) return { value: 0, isPositive: true };
    
    const firstPrice = predictions.predictions[0].predicted_price;
    const lastPrice = predictions.predictions[predictions.predictions.length - 1].predicted_price;
    const change = ((lastPrice - firstPrice) / firstPrice) * 100;
    
    return {
      value: Math.abs(change).toFixed(2),
      isPositive: change >= 0
    };
  };
  
  // Fetch data from APIs
  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      try {
        // Fetch all data in parallel
        const [modelInfoResponse, predictionsResponse, evaluationResponse] = await Promise.all([
          fetch('http://localhost:5000/api/model-info'),
          fetch('http://localhost:5000/api/predictions'),
          fetch('http://localhost:5000/api/evaluation')
        ]);
        
        if (!modelInfoResponse.ok || !predictionsResponse.ok || !evaluationResponse.ok) {
          throw new Error('Failed to fetch data from API');
        }
        
        const modelInfoData = await modelInfoResponse.json();
        const predictionsData = await predictionsResponse.json();
        const evaluationData = await evaluationResponse.json();
        
        setModelInfo(modelInfoData);
        setPredictions(predictionsData);
        setEvaluation(evaluationData);
      } catch (err) {
        setError(err.message);
        console.error("Error fetching data:", err);
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchData();
  }, []);
  
  // Handle loading state
  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-900">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-yellow-500 mx-auto"></div>
          <p className="mt-4 text-xl text-gray-300">Loading Bitcoin Price Predictor...</p>
        </div>
      </div>
    );
  }
  
  // Handle error state
  if (error) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-900">
        <div className="text-center text-red-500 p-8 rounded-lg border border-red-700 bg-red-900 bg-opacity-20">
          <h2 className="text-2xl font-bold mb-4">Error Loading Data</h2>
          <p>{error}</p>
          <p className="mt-4 text-sm">Please ensure the Flask backend is running on port 5000.</p>
        </div>
      </div>
    );
  }
  
  // Calculate price change
  const priceChange = calculatePriceChange();
  
  // Format date
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleString();
  };
  
  // Format data for prediction chart
  const chartData = predictions ? predictions.predictions.map(pred => ({
    hour: `H+${pred.hour}`,
    price: pred.predicted_price,
    lowerBound: pred.lower_bound,
    upperBound: pred.upper_bound
  })) : [];
  
  // Format data for model history chart
  const historyData = modelInfo ? modelInfo.train_history.loss.map((loss, idx) => ({
    epoch: idx + 1,
    trainLoss: loss,
    valLoss: modelInfo.train_history.val_loss[idx]
  })) : [];

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      {/* Header */}
      <header className="bg-gray-800 shadow-lg border-b border-yellow-700/30">
        <div className="container mx-auto px-4 py-6">
          <div className="flex justify-between items-center">
            <div className="flex items-center">
              <div className="bg-yellow-500 rounded-full p-2 mr-3">
                <TrendingUp className="h-6 w-6 text-gray-900" />
              </div>
              <div>
                <h1 className="text-2xl font-bold">Bitcoin Price Predictor</h1>
                <p className="text-gray-400 text-sm">Deep Learning-Powered Forecast</p>
              </div>
            </div>
            <div className="hidden md:block">
              <div className="bg-gray-700 rounded-lg px-4 py-2 flex items-center">
                <Clock className="h-4 w-4 mr-2 text-gray-400" />
                <span className="text-sm text-gray-300">
                  Last updated: {predictions ? formatDate(predictions.generated_at) : 'Loading...'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </header>
      
      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        {/* Current Prediction Card */}
        <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl shadow-xl border border-gray-700/50 p-6 mb-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="col-span-1 md:col-span-2">
              <h2 className="text-xl font-semibold mb-2 flex items-center">
                <TrendingUp className="h-5 w-5 mr-2 text-yellow-500" />
                Bitcoin Price Prediction (Next 24 Hours)
              </h2>
              <div className="h-64 md:h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                    <defs>
                      <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#EAB308" stopOpacity={0.8} />
                        <stop offset="95%" stopColor="#EAB308" stopOpacity={0} />
                      </linearGradient>
                      <linearGradient id="colorBounds" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#FEF3C7" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#FEF3C7" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="hour" stroke="#9CA3AF" />
                    <YAxis stroke="#9CA3AF" />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1F2937', borderColor: '#4B5563', borderRadius: '0.5rem' }}
                      formatter={(value) => [formatter.format(value), 'Price']}
                    />
                    <Area type="monotone" dataKey="upperBound" stroke="none" fillOpacity={1} fill="url(#colorBounds)" />
                    <Area type="monotone" dataKey="lowerBound" stroke="none" fillOpacity={1} fill="url(#colorBounds)" />
                    <Line 
                      type="monotone" 
                      dataKey="price" 
                      stroke="#EAB308" 
                      strokeWidth={2}
                      dot={{ r: 3, strokeWidth: 1 }}
                      activeDot={{ r: 5, strokeWidth: 2 }}
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </div>
            
            <div className="bg-gray-800/50 rounded-lg border border-gray-700/70 p-4">
              <h3 className="text-lg font-medium mb-4 text-center text-gray-300">Prediction Summary</h3>
              
              {predictions && (
                <>
                  <div className="mb-6">
                    <div className="text-sm text-gray-400 mb-1">Next Hour Price</div>
                    <div className="text-2xl font-bold text-white">
                      {formatter.format(predictions.predictions[0].predicted_price)}
                    </div>
                  </div>
                  
                  <div className="mb-6">
                    <div className="text-sm text-gray-400 mb-1">24 Hour Forecast</div>
                    <div className="text-2xl font-bold text-white">
                      {formatter.format(predictions.predictions[predictions.predictions.length - 1].predicted_price)}
                    </div>
                    <div className={`text-sm mt-1 flex items-center ${priceChange.isPositive ? 'text-green-500' : 'text-red-500'}`}>
                      {priceChange.isPositive ? (
                        <TrendingUp className="h-4 w-4 mr-1" />
                      ) : (
                        <TrendingUp className="h-4 w-4 mr-1 transform rotate-180" />
                      )}
                      {priceChange.value}% over 24h
                    </div>
                  </div>
                  
                  <div>
                    <div className="text-sm text-gray-400 mb-1">Model Confidence</div>
                    <div className="h-4 bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-yellow-600 to-yellow-400"
                        style={{ width: `${Math.min(modelInfo.metrics.short_term.directional_accuracy * 100, 100)}%` }}
                      ></div>
                    </div>
                    <div className="text-sm mt-1 text-gray-400 text-right">
                      {(modelInfo.metrics.short_term.directional_accuracy * 100).toFixed(1)}% directional accuracy
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
        
        {/* Model Info and Metrics */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Model Information */}
          <div className="bg-gray-800 rounded-lg border border-gray-700/50 shadow-lg">
            <div 
              className="px-6 py-4 flex justify-between items-center cursor-pointer hover:bg-gray-700/30"
              onClick={() => setShowDetails({...showDetails, model: !showDetails.model})}
            >
              <h2 className="text-lg font-semibold flex items-center">
                <Info className="h-5 w-5 mr-2 text-blue-400" />
                Model Information
              </h2>
              {showDetails.model ? <ChevronUp className="h-5 w-5" /> : <ChevronDown className="h-5 w-5" />}
            </div>
            
            {showDetails.model && modelInfo && (
              <div className="px-6 py-4 border-t border-gray-700/50">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h3 className="text-sm font-medium text-gray-400 mb-1">Architecture</h3>
                    <p className="text-base">{modelInfo.architecture}</p>
                  </div>
                  <div>
                    <h3 className="text-sm font-medium text-gray-400 mb-1">Training Date</h3>
                    <p className="text-base">{formatDate(modelInfo.training_date)}</p>
                  </div>
                  <div>
                    <h3 className="text-sm font-medium text-gray-400 mb-1">Look Back Window</h3>
                    <p className="text-base">{modelInfo.look_back} hours</p>
                  </div>
                  <div>
                    <h3 className="text-sm font-medium text-gray-400 mb-1">Forecast Horizon</h3>
                    <p className="text-base">{modelInfo.forecast_horizon} hours</p>
                  </div>
                </div>
                
                <div className="mt-4">
                  <h3 className="text-sm font-medium text-gray-400 mb-1">Training Progress</h3>
                  <div className="h-40">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={historyData} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis dataKey="epoch" stroke="#9CA3AF" />
                        <YAxis stroke="#9CA3AF" />
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#1F2937', borderColor: '#4B5563', borderRadius: '0.5rem' }}
                        />
                        <Legend />
                        <Line 
                          type="monotone" 
                          dataKey="trainLoss" 
                          name="Training Loss" 
                          stroke="#3B82F6" 
                          dot={false} 
                        />
                        <Line 
                          type="monotone" 
                          dataKey="valLoss" 
                          name="Validation Loss" 
                          stroke="#EC4899" 
                          dot={false} 
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                
                <div className="mt-4">
                  <h3 className="text-sm font-medium text-gray-400 mb-1">Features Used</h3>
                  <div className="flex flex-wrap gap-2 mt-2">
                    {modelInfo.features_used.map((feature, idx) => (
                      <span 
                        key={idx} 
                        className="px-2 py-1 bg-gray-700 text-xs rounded-full text-gray-300"
                      >
                        {feature}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
          
          {/* Performance Metrics */}
          <div className="bg-gray-800 rounded-lg border border-gray-700/50 shadow-lg">
            <div 
              className="px-6 py-4 flex justify-between items-center cursor-pointer hover:bg-gray-700/30"
              onClick={() => setShowDetails({...showDetails, metrics: !showDetails.metrics})}
            >
              <h2 className="text-lg font-semibold flex items-center">
                <BarChart2 className="h-5 w-5 mr-2 text-green-400" />
                Performance Metrics
              </h2>
              {showDetails.metrics ? <ChevronUp className="h-5 w-5" /> : <ChevronDown className="h-5 w-5" />}
            </div>
            
            {showDetails.metrics && modelInfo && (
              <div className="px-6 py-4 border-t border-gray-700/50">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Short-term metrics */}
                  <div className="bg-gray-700/30 p-4 rounded-lg">
                    <h3 className="text-md font-medium text-center mb-3 text-yellow-400">Short-term (1 hour)</h3>
                    <div className="space-y-3">
                      <div>
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-400">RMSE:</span>
                          <span className="font-medium">{modelInfo.metrics.short_term.rmse.toFixed(2)}</span>
                        </div>
                        <div className="h-2 bg-gray-700 rounded-full mt-1">
                          <div 
                            className="h-full bg-green-500 rounded-full" 
                            style={{ width: `${Math.max(100 - modelInfo.metrics.short_term.rmse / 10, 20)}%` }}
                          ></div>
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-400">MAPE:</span>
                          <span className="font-medium">{(modelInfo.metrics.short_term.mape * 100).toFixed(2)}%</span>
                        </div>
                        <div className="h-2 bg-gray-700 rounded-full mt-1">
                          <div 
                            className="h-full bg-blue-500 rounded-full" 
                            style={{ width: `${Math.max(100 - modelInfo.metrics.short_term.mape * 100, 20)}%` }}
                          ></div>
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-400">Directional Accuracy:</span>
                          <span className="font-medium">{(modelInfo.metrics.short_term.directional_accuracy * 100).toFixed(2)}%</span>
                        </div>
                        <div className="h-2 bg-gray-700 rounded-full mt-1">
                          <div 
                            className="h-full bg-yellow-500 rounded-full" 
                            style={{ width: `${modelInfo.metrics.short_term.directional_accuracy * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  {/* Long-term metrics */}
                  <div className="bg-gray-700/30 p-4 rounded-lg">
                    <h3 className="text-md font-medium text-center mb-3 text-yellow-400">Long-term (24 hours)</h3>
                    <div className="space-y-3">
                      <div>
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-400">RMSE:</span>
                          <span className="font-medium">
                            {modelInfo.metrics.long_term.rmse ? modelInfo.metrics.long_term.rmse.toFixed(2) : 'N/A'}
                          </span>
                        </div>
                        <div className="h-2 bg-gray-700 rounded-full mt-1">
                          <div 
                            className="h-full bg-green-500 rounded-full" 
                            style={{ width: `${modelInfo.metrics.long_term.rmse ? Math.max(100 - modelInfo.metrics.long_term.rmse / 10, 10) : 0}%` }}
                          ></div>
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-400">MAPE:</span>
                          <span className="font-medium">
                            {modelInfo.metrics.long_term.mape ? (modelInfo.metrics.long_term.mape * 100).toFixed(2) + '%' : 'N/A'}
                          </span>
                        </div>
                        <div className="h-2 bg-gray-700 rounded-full mt-1">
                          <div 
                            className="h-full bg-blue-500 rounded-full" 
                            style={{ width: `${modelInfo.metrics.long_term.mape ? Math.max(100 - modelInfo.metrics.long_term.mape * 100, 10) : 0}%` }}
                          ></div>
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-400">Directional Accuracy:</span>
                          <span className="font-medium">
                            {modelInfo.metrics.long_term.directional_accuracy ? (modelInfo.metrics.long_term.directional_accuracy * 100).toFixed(2) + '%' : 'N/A'}
                          </span>
                        </div>
                        <div className="h-2 bg-gray-700 rounded-full mt-1">
                          <div 
                            className="h-full bg-yellow-500 rounded-full" 
                            style={{ width: `${modelInfo.metrics.long_term.directional_accuracy ? modelInfo.metrics.long_term.directional_accuracy * 100 : 0}%` }}
                          ></div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="mt-4 p-3 bg-blue-900/20 border border-blue-800/30 rounded text-sm text-blue-300">
                  <p className="flex items-center">
                    <Info className="h-4 w-4 mr-2 flex-shrink-0" />
                    <span>
                      Lower RMSE and MAPE values indicate better accuracy. High directional accuracy means the model correctly predicts price movement direction.
                    </span>
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
        
        {/* Detailed Evaluation */}
        <div className="bg-gray-800 rounded-lg border border-gray-700/50 shadow-lg mb-8">
          <div 
            className="px-6 py-4 flex justify-between items-center cursor-pointer hover:bg-gray-700/30"
            onClick={() => setShowDetails({...showDetails, evaluation: !showDetails.evaluation})}
          >
            <h2 className="text-lg font-semibold flex items-center">
              <Award className="h-5 w-5 mr-2 text-purple-400" />
              Detailed Evaluation
            </h2>
            {showDetails.evaluation ? <ChevronUp className="h-5 w-5" /> : <ChevronDown className="h-5 w-5" />}
          </div>
          
          {showDetails.evaluation && evaluation && (
            <div className="px-6 py-4 border-t border-gray-700/50">
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-700">
                  <thead className="bg-gray-700/50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Horizon</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Dataset</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">RMSE</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">MAPE</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Directional Accuracy</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-700">
                    {Object.entries(evaluation.horizons).map(([horizon, data]) => (
                      <tr key={horizon}>
                        <td className="px-4 py-3 whitespace-nowrap text-sm font-medium text-yellow-400">{horizon} hour{horizon !== '1' ? 's' : ''}</td>
                        <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">Train</td>
                        <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">{data.train.rmse.toFixed(2)}</td>
                        <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">{(data.train.mape * 100).toFixed(2)}%</td>
                        <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">{(data.train.directional_accuracy * 100).toFixed(2)}%</td>
                      </tr>
                    ))}
                    {Object.entries(evaluation.horizons).map(([horizon, data]) => (
                      <tr key={`${horizon}-val`}>
                        <td className="px-4 py-3 whitespace-nowrap text-sm font-medium text-yellow-400"></td>
                        <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">Validation</td>
                        <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">{data.validation.rmse.toFixed(2)}</td>
                        <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">{(data.validation.mape * 100).toFixed(2)}%</td>
                        <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">{(data.validation.directional_accuracy * 100).toFixed(2)}%</td>
                      </tr>
                    ))}
                    {Object.entries(evaluation.horizons).map(([horizon, data]) => (
                      <tr key={`${horizon}-test`}>
                        <td className="px-4 py-3 whitespace-nowrap text-sm font-medium text-yellow-400"></td>
                        <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">Test</td>
                        <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">{data.test.rmse.toFixed(2)}</td>
                        <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">{(data.test.mape * 100).toFixed(2)}%</td>
                        <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">{(data.test.directional_accuracy * 100).toFixed(2)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              
              <div className="mt-4">
                <h3 className="text-sm font-medium text-gray-400 mb-2">Overfitting Analysis</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="bg-gray-700/30 p-3 rounded-lg">
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-400">Train to Validation Ratio:</span>
                      <span className="font-medium">{evaluation.overfitting_ratios.train_to_val.toFixed(2)}</span>
                    </div>
                    <div className="h-2 bg-gray-700 rounded-full">
                      <div 
                        className="h-full bg-purple-500 rounded-full" 
                        style={{ width: `${Math.min(evaluation.overfitting_ratios.train_to_val * 50, 100)}%` }}
                      ></div>
                    </div>
                    <p className="text-xs text-gray-400 mt-1">
                      Values close to 1 indicate good generalization
                    </p>
                  </div>
                  <div className="bg-gray-700/30 p-3 rounded-lg">
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-400">Train to Test Ratio:</span>
                      <span className="font-medium">{evaluation.overfitting_ratios.train_to_test.toFixed(2)}</span>
                    </div>
                    <div className="h-2 bg-gray-700 rounded-full">
                      <div 
                        className="h-full bg-indigo-500 rounded-full" 
                        style={{ width: `${Math.min(evaluation.overfitting_ratios.train_to_test * 50, 100)}%` }}
                      ></div>
                    </div>
                    <p className="text-xs text-gray-400 mt-1">
                      Values close to 1 indicate good generalization
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
        
        {/* Refresh Button */}
        <div className="flex justify-center">
          <button 
            onClick={() => window.location.reload()}
            className="flex items-center px-4 py-2 bg-yellow-600 hover:bg-yellow-700 text-white rounded-lg transition-colors duration-200"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh Predictions
          </button>
        </div>
      </main>
      
      {/* Footer */}
      <footer className="bg-gray-800 border-t border-gray-700/50 py-6">
        <div className="container mx-auto px-4 text-center text-gray-400 text-sm">
          <p>Bitcoin Price Predictor - Powered by Deep Learning (LSTM-GRU Hybrid Model)</p>
          <p className="mt-1">Data last updated: {modelInfo ? formatDate(modelInfo.last_data_timestamp) : 'Loading...'}</p>
        </div>
      </footer>
    </div>
  );
}