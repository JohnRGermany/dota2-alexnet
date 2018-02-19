# IMAGES
<table>
  <tr>
    <th>offset</th>
    <th>type</th>
    <th>value</th>
    <th>description</th>
  </tr>
  <tr>
    <td>0000</td>
    <td>32 bit integer</td>
    <td>0x646f7461</td>
    <td>magic number</td>
  </tr>
  <tr>
    <td>0004</td>
    <td>32 bit integer</td>
    <td>19999</td>
    <td>number of images</td>
  </tr>
  <tr>
    <td>0008</td>
    <td>32 bit integer</td>
    <td>160</td>
    <td>number of rows</td>
  </tr>
  <tr>
    <td>0012</td>
    <td>32 bit integer</td>
    <td>160</td>
    <td>number of cols</td>
  </tr>
  <tr>
    <td>0016</td>
    <td>unsigned byte</td>
    <td>??</td>
    <td>pixel</td>
  </tr>
  <tr>
    <td>0017</td>
    <td>unsigned byte</td>
    <td>??</td>
    <td>pixel</td>
  </tr>
  <tr>
    <td>XXXX</td>
    <td>unsigned byte</td>
    <td>??</td>
    <td>pixel</td>
  </tr>
</table>

# Labels
<table>
  <tr>
    <th>offset</th>
    <th>type</th>
    <th>value</th>
    <th>description</th>
  </tr>
  <tr>
    <td>0000</td>
    <td>32 bit integer</td>
    <td>0x646f7462</td>
    <td>magic number</td>
  </tr>
  <tr>
    <td>0004</td>
    <td>32 bit integer</td>
    <td>19999</td>
    <td>number of labels</td>
  </tr>
  <tr>
    <td>0008</td>
    <td>unsigned byte</td>
    <td>??</td>
    <td>label</td>
  </tr>
  <tr>
    <td>0017</td>
    <td>unsigned byte</td>
    <td>??</td>
    <td>label</td>
  </tr>
  <tr>
    <td>XXXX</td>
    <td>unsigned byte</td>
    <td>??</td>
    <td>label</td>
  </tr>
</table>
