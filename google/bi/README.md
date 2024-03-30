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

5. Now, we have some useful features and some are not here, I found out `ride_id` feature as an un-useful feature right now, let's remove it.

   ```python
   f(x) = Table.RemoveColumns(#"Changed Type",{"ride_id"})
   ```

![img](./output/img13.png)

6. Now, we can see here in our dataset there are two features `started_at` and `ended_at` but these two features having values as combination of data and time which we have to separate into `start_date`, `start_time`, `end_date` and `end_time` so, further we can derived new features from them such as `duration_of_ride` of something like that.

![img](./output/img14.png)

```python
f(x) = Table.DuplicateColumn(#"Duplicated Column1", "started_at", "started_at - Copy")
f(x) = Table.DuplicateColumn(#"Duplicated Column2", "started_at", "started_at - Copy.1")
f(x) = Table.DuplicateColumn(#"Duplicated Column1", "ended_at", "ended_at - Copy")
f(x) = Table.DuplicateColumn(#"Duplicated Column2", "ended_at", "ended_at - Copy.1")
```

![img](./output/img15.png)

7. Now, let's change the name of these columns and change their types

   ```python
   f(x) = Table.RenameColumns(#"Duplicated Column3",{{"started_at - Copy", "start_date"}})
   ```

![img](./output/img16.png)

```python
f(x) = Table.TransformColumnTypes(#"Renamed Columns",{{"start_date", type date}})
```

![img](./output/img17.png)

8. Change the name of another duplicated column to `start_hour` and then extract the hour values from it.

```python
f(x) = Table.RenameColumns(#"Extracted Time",{{"started_at - Copy.1", "start_hour"}})
```

then, extracting the hour values 

```python
f(x) = Table.TransformColumns(#"Changed Type1",{{"start_hour", Time.Hour, Int64.Type}})
```

![img](./output/img18.png)

9. Same the above operations we can perform for `end_date` and `end_hour` columns to generate, let's do it

![img](./output/img19.png)

10. Now, let's add a custom column and named it as `trip_duration_minutes` and it's basically, the difference between the `end_hour` and `start_hour` in minutes.

```python
f(x) = Table.AddColumn(#"Extracted Time1", "trip_duration_minutes", each [ended_at] - [started_at])
```

![img](./output/img20.png)

```python
f(x) = Table.TransformColumns(#"Added Custom",{{"trip_duration_minutes", Duration.Minutes, Int64.Type}})
```

![img]()
