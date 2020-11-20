---
layout: post
title: Test markdown
subtitle: Each post also has a subtitle
gh-repo: daattali/beautiful-jekyll
gh-badge: [star, fork, follow]
tags: [test]
comments: true
---

# Predicting Great Books from Goodreads Data Using Python

![books](https://images.unsplash.com/photo-1550399105-c4db5fb85c18?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1351&q=80)

## What
This is a data set of the first 50,000 book ids pulled from Goodreads' API on July 30th, 2020. A few thousand ids did not make it through because the book id was changed, the URL or API broke, or the information was stored in an atypical format.

## Why
From the reader's perspective, books are a multi-hour commitment of learning and leisure (they don't call it **Good**reads for nothing). From the author's and publisher's perspectives, books are a way of living (with some learning and leisure too). In both cases, knowing which factors explain and predict great books will save you time and money. Because while different people have different tastes and values, knowing how a book is rated in general is a sensible starting point. You can always update it later.

## Environment
It's good practice to work in a virtual environment, a sandbox with its own libraries and versions, so we'll make one for this project. There are several ways to do this, but we'll use [Anaconda](https://www.anaconda.com/products/individual). To create and activate an Anaconda virtual environment called 'gr' (for Goodreads) using Python 3.7, run the following commands in your terminal or command line:

```python
conda create -n gr python=3.7
conda activate gr
```

![books](https://images.unsplash.com/photo-1550399105-c4db5fb85c18?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1351&q=80)
Photo of old books by [Ed Robertson](https://unsplash.com/@eddrobertson) on [Unsplash](https://unsplash.com/)


const mediumToMarkdown = require('medium-to-markdown');
mediumToMarkdown.convertFromUrl('https://medium.com/@ryan.koul/predicting-great-books-from-goodreads-data-using-python-1d378e7ef926')
.then(function (markdown) {
  console.log(markdown); //=> Markdown content of medium post
}); >> _posts/2020-07-31-Predicting-Great-Books-from-Goodreads-Data-Using-Python.md


![Crepe](https://s3-media3.fl.yelpcdn.com/bphoto/cQ1Yoa75m2yUFFbY2xwuqw/348s.jpg){: .center-block :}

Here's a code chunk:

~~~
var foo = function(x) {
  return(x + 5);
}
foo(3)
~~~

And here is the same code with syntax highlighting:

```javascript
var foo = function(x) {
  return(x + 5);
}
foo(3)
```

And here is the same code yet again but with line numbers:

{% highlight javascript linenos %}
var foo = function(x) {
  return(x + 5);
}
foo(3)
{% endhighlight %}

## Boxes
You can add notification, warning and error boxes like this:

### Notification

{: .box-note}
**Note:** This is a notification box.

### Warning

{: .box-warning}
**Warning:** This is a warning box.

### Error

{: .box-error}
**Error:** This is an error box.
