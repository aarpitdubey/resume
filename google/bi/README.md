# GOOGLE BUSSINESS INTELLIGENCE PROJECT : Cyclistic Bike Share

## System Requirements

**Tools Used :**

- **Tableau :** Familarity with Tableau UI.
- **Excel :** Familarity with Excel. USE Power Query before.
- **Domain :** Having fundamental grasp of sales domain.

Dataset used:

As this dataset is more than 1GB so I'm going to upload it to : [Cyclistic Bike Share Dataset](https://drive.google.com/drive/folders/16MQJ6CQW-_tiZsoHRnMredfOzvulW_eA?usp=sharing)

## 1. Working with Excel - Power Query Editor.

1. Open your Ms Excel with new file or new sheet.
2. Then, click on `Data` tab shown in the image below
   ![excel](./output/img2.png)
3. Now you are able to see, multiple options like, from files, from databases, from Azure and other resources select from files and click it. Then, after it shows from workbook, from XML, from CSV , from Text and from Folder opthion select and click that last option.
   ![excel 2](./output/img3.png)
4. Now, you have the option to select the path of that folder or you can simply browse it from the button given there. Select the folder and then click ok!
   ![img](./output/img4.png)
5. After clicking the ok button, our power query editor will open with the files that folder containing
   ![img](./output/img5.png)

## Power Query Editor in Excel

1. For more and better understandings, I changed the `Query 1` file name to `bike_data` you can do it using the Query Settings option
   ![img](./output/img6.png)
2. We have 12 files and for all of them counted as rows and we have 8 columns created by the Power Query Editor
   ![img](./output/img7.png)
3. Here, we can see `Extention` feature so, to maintain the same or equal to a specific choice of extention containing files we can perform certain task to do that so. For that you have to click the down arrow as shown in extension feature

   ![img](./output/img8.png)

```python
f(x) = Table.SelectRows(Source, each [Extension] = ".csv")
```

![img](./output/img9.png)

```python
f(x) = Table.SelectColumns(#"Filtered Rows",{"Content"})
```

![img](./output/img10.png)

4. After removing other columns let's click on combine button shown in the below image.
   ![img](./output/img11.png)
   After clicking it the Power Query editor performed some operation automatically, and combine binaries those operations shown in the below image.

   ```python
   f(x) = Binary.Combine(#"Removed Other Columns"[Content]) # Combine binaries
   f(x) = Csv.Document(#"Combined Binaries",[Delimiter=",",Encoding=1252]) # Imported Csv
   f(x) = Table.PromoteHeaders(#"Imported CSV") # Promoted headers
   f(x) = Table.TransformColumnTypes(#"Promoted Headers",{{"ride_id", type text}, {"rideable_type", type text}, {"started_at", type datetime}, {"ended_at", type datetime}, {"start_station_name", type text}, {"start_station_id", type text}, {"end_station_name", type text}, {"end_station_id", type text}, {"start_lat", type number}, {"start_lng", type number}, {"end_lat", type number}, {"end_lng", type number}, {"member_casual", type text}}) # Changed Type
   ```

![img](./output/img12.png)

After clicking the combine button the Power Query performed 4 different functionalities:

- Combined Binaries
- Imported CSV
- Promoted Headers
- Changed Type
