import React from 'react';

function Header() {
  return (
    <header className="header">
      <div className="logo">
        <img src="/bitcoin-logo.svg" alt="Bitcoin" />
      </div>
      <h1>Bitcoin Price Prediction with LSTM</h1>
      <p>Deep Learning-based cryptocurrency price forecasting</p>
    </header>
  );
}

export default Header;