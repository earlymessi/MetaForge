# 📂 MetaForge Benchmark Datasets

MetaForge supports benchmark datasets for the **Job Shop Scheduling Problem (JSSP)** in multiple formats. This guide shows you:

- How to structure datasets
- Supported formats (`orlib` and `json`)
- How to add your own instances

---

## 📁 Dataset Folder

By default, benchmark files live in:

```
data/benchmarks/
```

You can change this folder when running:
```bash
python -m src.metaforge.utils.compare_solvers --benchmark_folder=my_folder/
```

---

## 📄 Supported Formats

### 1. OR-Library `.txt` Format (default)

Used by classic benchmark sets like *FT06*, *LA01*, *ORB05*, etc.

**Structure:**
```
num_jobs num_machines
m1 t1 m2 t2 ... mN tN
...
```

Each line is a job. Each job contains a sequence of machine-time pairs.

**Example:**
```
3 3
0 3 1 2 2 2
1 2 2 1 0 3
2 2 0 1 1 3
```

This means:
- Job 0 → [Machine 0: 3, Machine 1: 2, Machine 2: 2]
- Job 1 → ...
- Job 2 → ...

✅ File must end in `.txt`  
✅ No headers or extra whitespace

---

### 2. JSON Format (optional)

Use this if you want more flexibility (custom metadata, dynamic constraints, etc.)

**Structure:**
```json
{
  "jobs": [
    {
      "tasks": [
        { "machine_id": 0, "duration": 3 },
        { "machine_id": 1, "duration": 2 }
      ]
    },
    ...
  ],
  "metadata": {
    "source": "custom",
    "name": "DemoInstance"
  }
}
```

To load JSON, pass `format="json"` when calling `compare_all_benchmarks()`.

---

## ➕ Adding Your Own Benchmarks

Just drop `.txt` or `.json` files into `data/benchmarks/`

To test your instance quickly:
```python
from metaforge.problems.benchmark_loader import load_job_shop_instance
problem = load_job_shop_instance("data/benchmarks/my_instance.txt", format="orlib")
print(problem)
```

---

## ⚠️ Gotchas

- `.txt` only supports *strict* OR-Library format.
- `.json` allows you to add due dates, release times, etc. (coming soon in v2.0!)
- Use meaningful filenames: `ft06.txt`, `my_custom_instance.json`, etc.

---

## 📚 Benchmark Suggestions

You can start with classic OR-Library instances:

- `ft06.txt` (6 jobs, 6 machines)
- `la01.txt` to `la40.txt`
- `orb01.txt` to `orb10.txt`

These are already used in literature — perfect for testing and validation.

---

Happy benchmarking! 🧪
