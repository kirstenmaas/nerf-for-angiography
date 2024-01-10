import React, { useEffect, useState, useRef } from 'react';
import * as d3 from 'd3';
import _ from 'underscore';

import HeatmapRadial from './HeatmapRadial';
import d3EzBase from './d3EzBase';

import Legend from "../Legend/Legend";
import { interpolateCividis } from "https://cdn.skypack.dev/d3-scale-chromatic@3";

import { saveSvgAsPng } from 'save-svg-as-png';

export default function Heatmap(props) {

    const [ angleObject, setAngleObject ] = useState({
        'key': null,
        'order': null,
        'angles': [],
    })

    const [ dataObject, setDataObject ] = useState({});

    const { metric, direction, centerPoint, sparseAngle, limitedAngle, 
        sparsity, background, samplingStrategy, architecture, sortingObject, setSortingObject,
        showLabels } = props;

    const [ isLoaded, setLoaded ] = useState(false);
    const [ hasError, setError ] = useState(null);

    const domRef = useRef(null);
    const legendRef = useRef(null);

    const canvasPredRef = useRef();
    const canvasOrgRef = useRef();
    const canvasDiffRef = useRef();

    useEffect(() => {
        let newSortingObject = {
            'samplingStrategy': samplingStrategy,
            'sparsity': sparsity,
            'background': background,
            'limitedAngle': limitedAngle,
            'sparseAngle': sparseAngle,
            'centerPoint': centerPoint,
            'architecture': architecture,
            'metric': metric,
            'direction': direction,
            'x_axis': 'X',
            'y_axis': 'Z'
        }

        const newFileStr = getFetchString(newSortingObject, true);
        const oldFileStr = dataObject['fileStr'];

        if (_.isEmpty(dataObject) && domRef.current && !_.isEqual(newSortingObject, sortingObject)) {
            setSortingObject(newSortingObject);
            getDataset(newFileStr).then(
                (dataObj) => {Object.keys(dataObj).length > 0 && drawChart(dataObj, newSortingObject)});
        } else if (newFileStr !== oldFileStr || (!domRef.current && !hasError)) {
            setSortingObject(newSortingObject);
            getDataset(newFileStr).then(
                (dataObj) => {
                    if (!dataObj['error']) {
                        if (!dataObject['error']) {
                            d3.select(domRef.current).selectAll("*").remove();
                            d3.select(legendRef.current).selectAll("*").remove();
                        }
                        drawChart(dataObj, newSortingObject);
                    }
                });
        }

        const predCanvas = canvasPredRef.current;
        const orgCanvas = canvasOrgRef.current;
        const diffCanvas = canvasDiffRef.current;
        Object.keys(dataObject).length > 0 && angleObject['key'] !== null && drawImages(predCanvas, orgCanvas, diffCanvas);
    }, [dataObject, metric, centerPoint, direction, sparseAngle, limitedAngle, angleObject, background, sparsity, samplingStrategy, architecture, hasError])

    function getFetchString(sortingObject, fetchMetrics) {
        let fileStr = `http://localhost:8080`;
        let folderName = '/';
        let subFolderName = `/${sortingObject['limitedAngle']}-${sortingObject['sparseAngle']}-${sortingObject['centerPoint']}/`;

        const samplingStrategy = sortingObject['samplingStrategy'];
        const background = sortingObject['background'];
        const sparsity = sortingObject['sparsity'];
        const architecture = sortingObject['architecture'];

        if ( samplingStrategy === 'frangi' && architecture == '4x128' ) {
            if (!background && sparsity === 'ct') {
                folderName += 'limited-sparse-ct';
            } else if (background && sparsity === 'ct') {
                folderName += 'background-ct'
            } else if ( sparsity === 'lca' ) {
                folderName += 'sparsity-lca'
            }
        } else {
            if (architecture !== '4x128') {
                folderName += `architecture-${architecture}-${sparsity}`
            } else if (!background) {
                folderName += `sparsity-${samplingStrategy}-${sparsity}`
            } else {
                folderName += `background-${samplingStrategy}-${sparsity}`
            }
        }

        fileStr += folderName;
        fileStr += subFolderName;
        if (fetchMetrics) {
            fileStr += `${sortingObject['metric']}-${sortingObject['direction']}-${sortingObject['x_axis']}-${sortingObject['y_axis']}.json`;
        } else {
            fileStr += `theta-${angleObject.angles[0]}.0.json`
        }

        console.log(fileStr);

        return fileStr;
    }

    function drawImages(predCanvas, orgCanvas, diffCanvas) {
        let orgWidth = 100;
        let orgHeight = 100;
        if (sortingObject['sparsity'] === 'lca') {
            orgWidth = 150;
            orgHeight = 162;
        }

        const canvasWidth = predCanvas.width;
        const canvasHeight = predCanvas.height;

        const predCtx = predCanvas.getContext('2d');
        const orgCtx = orgCanvas.getContext('2d');
        const diffCtx = diffCanvas.getContext('2d');

        // get image from respective file
        const fileStr = getFetchString(sortingObject, false)
        fetch(fileStr)
            .then(res => res.json())
            .then((json_imgs) => {
                const phiIndex = json_imgs['phi'].indexOf(angleObject.angles[1]);

                const predImage = json_imgs['pred'][phiIndex];
                const orgImage = json_imgs['org'][phiIndex];
                const diffImage = json_imgs['diff'][phiIndex];

                // convert to rgba
                let j = 0;
                let stepSize = 4;

                let predDataImage = predCtx.createImageData(orgWidth, orgHeight);
                let orgDataImage = orgCtx.createImageData(orgWidth, orgHeight);
                let diffDataImage = diffCtx.createImageData(orgWidth, orgHeight);

                for (let i = 0; i < predImage.length; i++) {
                    for (let count = 0; count < stepSize - 1; count++) {
                        predDataImage.data[j+count] = 255 * predImage[i];
                        orgDataImage.data[j+count] =  255 * orgImage[i];
                        diffDataImage.data[j+count] = 255 * diffImage[i];
                    }

                    predDataImage.data[j+3] = 255;
                    orgDataImage.data[j+3] =  255;
                    diffDataImage.data[j+3] = 255;

                    j += stepSize;
                }

                createImageBitmap(predDataImage).then(renderer => {
                    predCtx.drawImage(renderer, 0,0, canvasWidth, canvasHeight)
                })
                createImageBitmap(orgDataImage).then(renderer => 
                    orgCtx.drawImage(renderer, 0,0, canvasWidth, canvasHeight)
                )
                createImageBitmap(diffDataImage).then(renderer => 
                    diffCtx.drawImage(renderer, 0,0, canvasWidth, canvasHeight)
                )
            },
            (error) => {
                setLoaded(true);
                setError(error);
            })
    }

    function drawChart(dataObject, sortingObject) {
        let thresholds = [10, 50];
        if (metric === 'SSIM') {
            thresholds = [0.92, 1]
            if (sortingObject['sparsity'] === 'lca') {
                thresholds = [0.7, 1]
            } else if (sortingObject['background']) {
                thresholds = [0.3, 1]
            }
        } else if (metric === 'DICE 2D') {
            thresholds = [0.8, 1]
        }

        var size = 800;

        const ownColorScale = d3.scaleSequential(interpolateCividis)
				.domain(thresholds);

        Legend(ownColorScale, legendRef, {
            title: `${sortingObject.metric}`,
            width: size,
        })

        let chart = null;
        if ( showLabels ) {
            chart = HeatmapRadial()
            .colorScale(ownColorScale)
            .sectorLabels(dataObject['sectorLabels'])
            .ringLabels(dataObject['ringLabels'])
            .thresholds(thresholds)
            .innerRadius(5);
        } else {
            chart = HeatmapRadial()
            .colorScale(ownColorScale)
            .thresholds(thresholds)
            .innerRadius(5);
        }

        // Create chart base
        var myChart = d3EzBase()
            .width(size)
            .height(size)
            .chart(chart)
            // .title(title)
            .on("customValueMouseOver", (d) => {
                    setAngleObject(d.data);
                }
            )

        d3.select(domRef.current)
            .datum(dataObject['data'])
            .call(myChart)
            .on('click', function() {
                const chartSvg = d3.select(domRef.current).select('svg')._groups[0][0];
                const fileStr = `${sortingObject['limitedAngle']}-${sortingObject['sparseAngle']}.png`;
                saveSvgAsPng(chartSvg, fileStr);
            });
        
    }
    

    async function getDataset(fileStr) {
        let dataObj = {};
        await fetch(fileStr, { method: 'GET' })
            .then(res => res.json())
            .then(
                (json) => {
                    setError(null);

                    var thetas = json['theta'];
                    var rads = json['rad']
                    var vals = json['vals'];
                    var angles = json['angles'];
    
                    let data = [];
                    const uniqueThetas = [...new Set(thetas)];
                    const uniqPosThetas = uniqueThetas.map(v => v < 0 ? Math.PI - v : v );
                    
                    let orderDict = {};
                    uniqueThetas.forEach((key, i) => orderDict[key] = uniqPosThetas[i]);
    
                    const firstQuarterThetas = uniqueThetas.filter(v => v <= Math.PI / 2 && v >= 0);
    
                    let ringLabels = {}
                    let sectorLabels = {}
    
                    for (let i = 0; i < thetas.length; i++) {
                        let theta = thetas[i];
                        let rad = rads[i];
                        let angle = angles[i]
    
                        if (rad === 0) {
                            // fix incorrect theta values
                            const filtered = Object.fromEntries(Object.entries(sectorLabels).filter(([k,v]) => v === angle[1]));
                            theta = Object.keys(filtered)[0] ? Number(Object.keys(filtered)[0]) : theta;
                        }
    
                        let order = orderDict[theta];
                        if (theta >= 0 && firstQuarterThetas.includes(theta)) {
                            order = order + 2*Math.PI
                            // order = firstQuarterThetas.indexOf(theta) + uniqueThetas.length; 
                        }
    
                        let newObject = { key: theta, order: order, value: vals[i], angles: angle };
                        if (!data.map(a=>a.key).includes(rad)) {
                            // let keyObject = { key: thetas[i], values: [newObject] }
                            let keyObject = { key: rad, values: [newObject] }
                            data.push(keyObject);
    
                            ringLabels[rad] = angle[0];
                        } else {
                            let keyObject = data.filter(a => a.key === rad)[0];
    
                            let keyIndex = data.findIndex(a => a.key === rad);
                            
                            let values = keyObject.values;
    
                            if (!values.map(a=>a.key).includes(theta)) {
                                values.push(newObject);
                                // reorder values such that 0° is on x axis
                                values = values.sort((a,b) => a.order < b.order ? 1 : -1);
                                keyObject[keyIndex] = values;
    
                                sectorLabels[theta] = angle[1];
                            }
    
                            ringLabels[rad] = angle[0] < ringLabels[rad] && angle[0] >= 0 ? angle[0] : ringLabels[rad];
                            //Math.min(angle[0], ringLabels[rad]);
                        }
                    }
    
                    if (data[0].values.length !== data[data.length - 1].values.length) {
                        let newDataValues = [];
    
                        const realValues = data[0].values;
                        const oldValues = data[data.length - 1].values;
    
                        for (var keyVal in realValues) {
                            const phiAngle = realValues[keyVal]['angles'][1];
                            const newKey = realValues[keyVal]['key'];
                            const newOrder = realValues[keyVal]['order'];
                            // check if key already saved
                            const currentValObject = oldValues.filter(v => v.key === newKey);
    
                            if (currentValObject.length === 1) {
                                newDataValues.push(currentValObject[0])
                            } else {
    
                                let newObjects = oldValues.filter(v => v.angles[1] === phiAngle);
                                if (newObjects.length === 1) {
                                    let newObject = newObjects[0];
    
                                    newObject['key'] = newKey;
                                    newObject['order'] = newOrder;
    
                                    newDataValues.push(newObject)
                                }
                            }
                        }
                        data[data.length - 1].values = newDataValues;
                    }

                    dataObj = { 'fileStr': fileStr, 'data': data, 'ringLabels': ringLabels, 'sectorLabels': sectorLabels };

                    Promise.resolve()
                        .then(() => { setDataObject(dataObj) })
                        .then(() => { setLoaded(true) })
                        .then(() => { return dataObj })
                },
                (error) => {
                    dataObj['fileStr'] = fileStr;
                    dataObj['error'] = true;

                    setDataObject(dataObj);
                    setLoaded(true);
                    setError(error);
                }
            )
        return dataObj;
    }

    return (
        <div style={{ display: 'flex', margin: '2vw' }}>
            <div>
                <div style={{ display: 'flex', minWidth: '35vw', width: '40vw', flexWrap: 'wrap', gap: '1vw', margin: '0 1vw' }}>
                    <div
                        style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', minWidth: '300px'}}>
                        {typeof angleObject.angles[0] !== 'undefined' && <span style={{ fontSize: '20px', margin: '0.5vw 0' }}>{angleObject.angles[0] + '\u00B0 \u03B8'}</span>}
                        {typeof angleObject.angles[1] !== 'undefined' && <span style={{ fontSize: '20px', margin: '0.5vw 0' }}>{angleObject.angles[1] + '\u00B0 \u03C6'}</span>}
                    </div>
                    {[{'Difference': canvasDiffRef}, {'Prediction': canvasPredRef}, {'Original': canvasOrgRef}].map((canvasDict, i) => {
                        let canvasKey = Object.keys(canvasDict)[0];
                        return (
                            <div key={canvasKey} style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center'}}>
                                {canvasKey}
                                <canvas 
                                    ref={canvasDict[canvasKey]} 
                                    width='300' 
                                    height='300' 
                                    style={{ margin: '0.5vw 0'}}
                                />
                            </div>
                        )
                    })}
                    
                </div>
                
            </div>
            { hasError ? 
                <span>Could not load heatmap {String(hasError)}</span> : 
                !isLoaded ? 
                    <span>Loading...</span> :
                    <div>
                        <div ref={domRef} /> 
                        <svg ref={legendRef}
                            width='1000'
                        />
                    </div>
            }
        </div>
    )
}