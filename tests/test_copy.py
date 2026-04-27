def test_deep_copy_independence(brighton_problem):
    obj1 = brighton_problem
    obj2 = obj1.copy()

    # Modify the copy
    obj2._verbose = False

    # Original should NOT change
    assert obj1._verbose

    # Ensure they are not the same object
    assert obj1 is not obj2
