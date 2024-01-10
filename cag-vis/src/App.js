import { Paper } from '@mui/material';
import React, { useState } from 'react';

import Heatmap from "./components/Heatmap/ReactHeatmap";
import Options from "./components/Options";

import './App.css';

function App() {
  const [ metric, setMetric ] = useState('SSIM');
  const [ direction, setDirection ] = useState('top');
  const [ centerPoint, setCenterPoint ] = useState('[90  0]');
  const [ sparseAngle, setSparseAngles ] = useState(6);
  const [ limitedAngle, setLimitedAngles ] = useState(180);
  const [ xAxis, setXAxis ] = useState('X');
  const [ yAxis, setYAxis ] = useState('Z');
  const [ sparsity, setSparsity ] = useState('ct');
  const [ background, setBackground ] = useState(false);
  const [ samplingStrategy, setSamplingStrategy ] = useState('frangi');
  const [ architecture, setArchitecture ] = useState('4x128');

  const [ sortingObject, setSortingObject ] = useState(null);

  const [ showLabels, setShowLabels ] = useState(false);

  return (
    <div style={{ margin: '2vw'}}>
      <Paper variant='outlined'>
        <Options
          metric={metric}
          setMetric={setMetric}
          direction={direction}
          setDirection={setDirection}
          centerPoint={centerPoint}
          setCenterPoint={setCenterPoint}
          sparseAngle={sparseAngle}
          setSparseAngles={setSparseAngles}
          limitedAngle={limitedAngle}
          setLimitedAngles={setLimitedAngles}
          xAxis={xAxis}
          setXAxis={setXAxis}
          yAxis={yAxis}
          setYAxis={setYAxis}
          sparsity={sparsity}
          setSparsity={setSparsity}
          background={background}
          setBackground={setBackground}
          samplingStrategy={samplingStrategy}
          setSamplingStrategy={setSamplingStrategy}
          architecture={architecture}
          setArchitecture={setArchitecture}
          sortingObject={sortingObject}
        />
      </Paper>
      <Heatmap 
        metric={metric}
        direction={direction}
        centerPoint={centerPoint}
        sparseAngle={sparseAngle}
        limitedAngle={limitedAngle}
        showLabels={showLabels}
        sparsity={sparsity}
        background={background}
        samplingStrategy={samplingStrategy}
        architecture={architecture}
        sortingObject={sortingObject}
        setSortingObject={setSortingObject}
      />
    </div>
  );
}

export default App;
