## Cause

The problem is in `weather/weatheragent.py`: you are starting uvicorn with an app object while also enabling reload:

```python
uvicorn.run(app, host="127.0.0.1", port=8001, reload=True)
```

Uvicorn explicitly requires an import string when `reload=True` or `workers` is used, such as:

```python
uvicorn.run("weatheragent:app", reload=True)
```

## Correct ways to run it

### Option 1: Best for reload

From the folder that contains the file:

```powershell
cd C:\Users\yienc\projects\stream-agent\weather
python -m uvicorn weatheragent:app --host 127.0.0.1 --port 8001 --reload
```

This works because `weatheragent` is importable as a module name in that directory.

### Option 2: Run the file directly without reload

If you want to keep it as a script, remove `reload=True`:

```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
```

Then run:

```powershell
cd C:\Users\yienc\projects\stream-agent\weather
python .\weatheragent.py
```

## Recommended fix for your file

Change this in `weather/weatheragent.py`:

```python
uvicorn.run(app, host="127.0.0.1", port=8001, reload=True)
```

to either:

```python
uvicorn.run("weatheragent:app", host="127.0.0.1", port=8001, reload=True)
```

or remove `reload=True` if you just want a normal server.

> The key rule is: `reload=True` requires an import string, not the app object itself.
