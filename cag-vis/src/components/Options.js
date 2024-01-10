import React from 'react';

import { Radio, FormControl, FormLabel, RadioGroup, FormControlLabel, Box, Slider, Typography } from '@mui/material';

export default function Options(props) {
    const { metric, setMetric, 
        direction, setDirection, 
        limitedAngle, setLimitedAngles, 
        sparseAngle, setSparseAngles,
        centerPoint, setCenterPoint,
        xAxis, yAxis, setXAxis, setYAxis,
        sparsity, setSparsity,
        background, setBackground,
        samplingStrategy, setSamplingStrategy,
        architecture, setArchitecture,
        sortingObject,
    } = props;

    const limitedAngleMarks = [
        { value: 5, label: "5\u00B0" },
        { value: 15, label: "15\u00B0" },
        { value: 30, label: "30\u00B0" },
        { value: 45, label: "45\u00B0" },
        { value: 60, label: "60\u00B0" },
        { value: 90, label: "90\u00B0" },
        { value: 180, label: "180\u00B0" },
    ]

    const sparseAngleMarks = [
        { value: 1, label: "4" },
        { value: 2, label: "9" },
        { value: 3, label: "16" },
        { value: 4, label: "25" },
        { value: 5, label: "36" },
        { value: 6, label: "49" },
    ]

    const limited180SparseAngleMarks = sparseAngleMarks.filter(m => m.value !== 1);
    
    return (
        <div style={{ display: 'flex', columnGap: '3vw', padding: '1vw' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0vw' }}>
                <div>
                    <Typography gutterBottom fontStyle={{ fontWeight: '500' }}>
                        Limited projections
                    </Typography>
                    <Box sx={{ width: 500 }}>
                        <Slider
                            aria-label="Restricted values"
                            min={5}
                            max={180}
                            value={limitedAngle}
                            onChange={(event) => setLimitedAngles(event.target.value)}
                            step={null}
                            marks={limitedAngleMarks}
                        />
                    </Box>
                </div>
                <div>
                    <Typography gutterBottom fontStyle={{ fontWeight: '500' }}>
                        Sparse projections
                    </Typography>
                    <Box sx={{ width: 500 }}>
                        <Slider
                            aria-label="Restricted values"
                            value={sparseAngle}
                            min={1}
                            max={6}
                            onChange={(event) => setSparseAngles(event.target.value)}
                            step={null}
                            marks={sortingObject && sortingObject['limitedAngle'] === 180 ? limited180SparseAngleMarks : sparseAngleMarks}
                        />
                    </Box>
                </div>
            </div>
            <FormControl>
                <FormLabel id="demo-radio-buttons-group-label" style={{ color: 'black', fontWeight: '500' }}>Sparsity</FormLabel>
                <RadioGroup
                    aria-labelledby="demo-radio-buttons-group-label"
                    name="radio-buttons-group"
                    value={sparsity}
                    onChange={(event) => {
                        setSparsity(event.target.value);
                        if (event.target.value === 'lca') {
                            setBackground(false);
                        }
                    }}
                >
                    <FormControlLabel value="ct" control={<Radio />} label="Low" />
                    <FormControlLabel value="lca" control={<Radio />} label="High" />
                </RadioGroup>
            </FormControl>
            <FormControl>
                <FormLabel id="demo-radio-buttons-group-label" style={{ color: 'black', fontWeight: '500' }}>Background</FormLabel>
                <RadioGroup
                    aria-labelledby="demo-radio-buttons-group-label"
                    name="radio-buttons-group"
                    value={background}
                    onChange={(event) => setBackground(event.target.value === 'true')}
                >
                    <FormControlLabel value={false} control={<Radio />} label="Binary" />
                    <FormControlLabel value={true} control={<Radio />} label="Background" disabled={sortingObject && sortingObject['sparsity'] === 'lca'} />
                </RadioGroup>
            </FormControl>
            <FormControl>
                <FormLabel id="demo-radio-buttons-group-label" style={{ color: 'black', fontWeight: '500' }}>Sampling strategy</FormLabel>
                <RadioGroup
                    aria-labelledby="demo-radio-buttons-group-label"
                    name="radio-buttons-group"
                    value={samplingStrategy}
                    onChange={(event) => setSamplingStrategy(event.target.value)}
                >
                    <FormControlLabel value="frangi" control={<Radio />} label="Frangi" />
                    <FormControlLabel value="segmentation" control={<Radio />} label="Segmentation" />
                    <FormControlLabel value='random' control={<Radio />} label="Random" />
                </RadioGroup>
            </FormControl>
            <FormControl>
                <FormLabel id="demo-radio-buttons-group-label" style={{ color: 'black', fontWeight: '500' }}>Model architecture</FormLabel>
                <RadioGroup
                    aria-labelledby="demo-radio-buttons-group-label"
                    name="radio-buttons-group"
                    value={architecture}
                    onChange={(event) => setArchitecture(event.target.value)}
                >
                    <FormControlLabel value='4x128' control={<Radio />} label="4x128" />
                    <FormControlLabel value='2x128' control={<Radio />} label="2x128" />
                    <FormControlLabel value='4x64' control={<Radio />} label="4x64" />
                </RadioGroup>
            </FormControl>
            <FormControl>
                <FormLabel id="demo-radio-buttons-group-label" style={{ color: 'black', fontWeight: '500' }}>Metric</FormLabel>
                <RadioGroup
                    aria-labelledby="demo-radio-buttons-group-label"
                    name="radio-buttons-group"
                    value={metric}
                    onChange={(event) => setMetric(event.target.value)}
                >
                    <FormControlLabel value="SSIM" control={<Radio />} label="SSIM" />
                    <FormControlLabel value="PSNR" control={<Radio />} label="PSNR" />
                    <FormControlLabel value='DICE 2D' control={<Radio />} label="DICE" />
                </RadioGroup>
            </FormControl>
            <FormControl>
                <FormLabel id="demo-radio-buttons-group-label" style={{ color: 'black', fontWeight: '500' }}>Direction</FormLabel>
                <RadioGroup
                    aria-labelledby="demo-radio-buttons-group-label"
                    name="radio-buttons-group"
                    value={direction}
                    onChange={(event) => setDirection(event.target.value)}
                >
                    <FormControlLabel value="top" control={<Radio />} label="Top" />
                    <FormControlLabel value="bottom" control={<Radio />} label="Bottom" />
                </RadioGroup>
            </FormControl>
            <FormControl disabled>
                <FormLabel id="demo-radio-buttons-group-label" style={{ color: 'black', fontWeight: '500' }}>Centerpoint</FormLabel>
                <RadioGroup
                    aria-labelledby="demo-radio-buttons-group-label"
                    name="radio-buttons-group"
                    value={centerPoint}
                    onChange={(event) => setCenterPoint(event.target.value)}
                >
                    <FormControlLabel value="[90  0]" control={<Radio />} label="(90, 0)" />
                    <FormControlLabel value="[0 0]" control={<Radio />} label="(0, 0)" />
                    <FormControlLabel value="[ 0 90]" control={<Radio />} label="(0, 90)" />
                </RadioGroup>
            </FormControl>
            <FormControl disabled>
                <FormLabel id="demo-radio-buttons-group-label" style={{ color: 'black', fontWeight: '500' }}>Axes</FormLabel>
                <RadioGroup
                    aria-labelledby="demo-radio-buttons-group-label"
                    name="radio-buttons-group"
                    value={`[${xAxis}, ${yAxis}]`}
                    onChange={(event) => {
                        let values = event.target.value.split(',')
                        setXAxis(values[0][-1]);
                        setYAxis(values[1][1])
                    }}
                >
                    <FormControlLabel value='[X, Y]' control={<Radio />} label="(X, Y)" />
                    <FormControlLabel value='[X, Z]' control={<Radio />} label="(X, Z)" />
                    <FormControlLabel value='[Y, Z]' control={<Radio />} label="(Y, Z)" />
                </RadioGroup>
            </FormControl>
        </div>
    )
}